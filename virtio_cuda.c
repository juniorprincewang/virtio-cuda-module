/*
 * Copyright (C) 2006, 2007, 2009 Rusty Russell, IBM Corporation
 * Copyright (C) 2009, 2010, 2011 Red Hat, Inc.
 * Copyright (C) 2009, 2010, 2011 Amit Shah <amit.shah@redhat.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */
#include <linux/cdev.h>
#include <linux/debugfs.h>
#include <linux/completion.h>
#include <linux/device.h>
#include <linux/err.h>
#include <linux/freezer.h>
#include <linux/fs.h>
#include <linux/splice.h>
#include <linux/pagemap.h>
#include <linux/init.h>
#include <linux/list.h>
#include <linux/poll.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/virtio.h>
#include <linux/wait.h>
#include <linux/workqueue.h>
#include <linux/module.h>
#include <linux/dma-mapping.h>

#include <linux/proc_fs.h> // proc operations PDE_DATA
#include "virtio-ioc.h"
#include "virtio_cuda.h"

#define ARG_SIZE sizeof(VirtIOArg)
#define VIRTIO_INDIRECT_NUM_MAX 1000

#ifdef VIRTIO_CUDA_DEBUG
#define gldebug(fmt, arg...) printk(KERN_DEBUG fmt, ##arg)
#define func() pr_info("[FUNC]%s\n",__FUNCTION__)

#else
#define gldebug(fmt, arg...) 
#define func() 
#endif

#define is_rproc_enabled IS_ENABLED(CONFIG_REMOTEPROC)

typedef struct MemObjectList {
	uint64_t addr;
	size_t size;
	struct list_head list;
} MOL;


/*
 * This is a global struct for storing common data for all the devices
 * this driver handles.
 *
 * Mainly, it has a linked list for all the consoles in one place so
 * that callbacks from hvc for get_chars(), put_chars() work properly
 * across multiple devices and multiple ports per device.
 */
struct ports_driver_data {
	/* Used for registering chardevs */
	struct class *class;

	/* Used for exporting per-port information to debugfs */
	struct dentry *debugfs_dir;

	/* Used for exporting portdev information, including virt_dev_count */
	struct proc_dir_entry *proc_dir;

	/* List of all the devices we're handling */
	struct list_head portdevs;

	/*
	 * This is used to keep track of the number of hvc consoles
	 * spawned by this driver.  This number is given as the first
	 * argument to hvc_alloc().  To correctly map an initial
	 * console spawned via hvc_instantiate to the console being
	 * hooked up via hvc_alloc, we need to pass the same vtermno.
	 *
	 * We also just assume the first console being initialised was
	 * the first one that got used as the initial console.
	 */
	unsigned int next_vtermno;

	/* All the console devices handled by this driver */
	struct list_head consoles;
};
static struct ports_driver_data pdrvdata;

static DEFINE_SPINLOCK(pdrvdata_lock);
static DECLARE_COMPLETION(early_console_added);

/* This struct holds information that's relevant only for console ports */
struct console {
	/* We'll place all consoles in a list in the pdrvdata struct */
	struct list_head list;

	/* The hvc device associated with this console port */
	struct hvc_struct *hvc;

	/*
	 * This number identifies the number that we used to register
	 * with hvc in hvc_instantiate() and hvc_alloc(); this is the
	 * number passed on by the hvc callbacks to us to
	 * differentiate between the other console ports handled by
	 * this driver
	 */
	u32 vtermno;
};

struct port_buffer {
	char *buf;

	/* size of the buffer in *buf above */
	size_t size;

	/* used length of the buffer */
	size_t len;
	/* offset in the buf from which to consume data */
	size_t offset;

	/* DMA address of buffer */
	dma_addr_t dma;

	/* Device we got DMA memory from */
	struct device *dev;

	/* List of pending dma buffers to free */
	struct list_head list;

	/* If sgpages == 0 then buf is used */
	unsigned int sgpages;

	/* sg is used if spages > 0. sg must be the last in is struct */
	struct scatterlist sg[0];
};

/*
 * virtual GPU device information on hosts, they are lists in 
 * struct ports_device.
 * Because we cann't use cuda.h or cuda_runtime.h in kernel, we store
 * any structure of CUDA by buf and size.
 */
struct vgpu_device {
	/* Next vgpu in the list, head is in the ports_device */
	struct list_head list;
	/* The 'id' to identify the gpu with the Host */
	u32 id;
	/* device flags */
	unsigned int flags;
	/* is device initialized */
	int initialized;
	/* sizeof struct cudaDeviceProp*/
	u32 prop_size;
	/* buf of struct cudaDeviceProp*/
	char *prop_buf;
};

/*
 * This is a per-device struct that stores data common to all the
 * ports for that device (vdev->priv).
 */
struct ports_device {
	/* Next portdev in the list, head is in the pdrvdata struct */
	struct list_head list;

	/*
	 * Workqueue handlers where we process deferred work after
	 * notification
	 */
	struct work_struct control_work;
	struct work_struct config_work;

	struct list_head ports;
	struct list_head vgpus;

	/* number of vgpus host holds */
	u32 nr_vgpus;

	/* To protect the list of ports */
	spinlock_t ports_lock;
	/* To protect the list of vgpus */
	spinlock_t vgpus_lock;

	/* To protect the vq operations for the control channel */
	spinlock_t c_ivq_lock;
	spinlock_t c_ovq_lock;

	/* max. number of ports this device can hold */
	u32 max_nr_ports;

	/* number of ports this device holds */
	u32 nr_ports;

	/* The virtio device we're associated with */
	struct virtio_device *vdev;

	/*
	 * A couple of virtqueues for the control channel: one for
	 * guest->host transfers, one for host->guest transfers
	 */
	struct virtqueue *c_ivq, *c_ovq;

	/*
	 * A control packet buffer for guest->host requests, protected
	 * by c_ovq_lock.
	 */
	struct virtio_console_control cpkt;

	/* Array of per-port IO virtqueues */
	struct virtqueue **in_vqs, **out_vqs;

	/* Major number for this device.  Ports will be created as minors. */
	int chr_major;

	/* File in the proc directory that exposes this portdev's information */
	struct proc_dir_entry *proc_virt_dev_count;
};

struct port_stats {
	unsigned long bytes_sent, bytes_received, bytes_discarded;
};

/* kernel memory for mmapping*/
struct virtio_uvm_page{
	unsigned long uvm_start;
	unsigned long uvm_end;
	struct list_head list;
	struct sg_table *st;
};

/* This struct holds the per-port data */
struct port {
	/* Next port in the list, head is in the ports_device */
	struct list_head list;

	/* Pointer to the parent virtio_console device */
	struct ports_device *portdev;

	/* The current buffer from which data has to be fed to readers */
	struct port_buffer *inbuf;

	/* list for struct virtio_uvm_page*/
	struct list_head page;
	/*
	 * To protect the operations on the in_vq associated with this
	 * port.  Has to be a spinlock because it can be called from
	 * interrupt context (get_char()).
	 */
	spinlock_t inbuf_lock;

	/* Protect the operations on the out_vq. */
	spinlock_t outvq_lock;

	/* The IO vqs for this port */
	struct virtqueue *in_vq, *out_vq;

	/* File in the debugfs directory that exposes this port's information */
	struct dentry *debugfs_file;

	/*
	 * Keep count of the bytes sent, received and discarded for
	 * this port for accounting and debugging purposes.  These
	 * counts are not reset across port open / close events.
	 */
	struct port_stats stats;

	/*
	 * The entries in this struct will be valid if this port is
	 * hooked up to an hvc console
	 */
	struct console cons;

	/* Each port associates with a separate char device */
	struct cdev *cdev;
	struct device *dev;

	/* Reference-counting to handle port hot-unplugs and file operations */
	struct kref kref;

	/* A waitqueue for poll() or blocking read operations */
	wait_queue_head_t waitqueue;

	/* The 'name' of the port that we expose via sysfs properties */
	char *name;

	/* We can notify apps of host connect / disconnect events via SIGIO */
	struct fasync_struct *async_queue;

	/* The 'id' to identify the port with the Host */
	u32 id;

	bool outvq_full;

	/* Is the host device open */
	bool host_connected;

	/* We should allow only one process to open a port */
	bool guest_connected;

	/* device memory list*/
	struct list_head device_mem_list;

	/* guest malloc memory list*/
	struct list_head guest_mem_list;

	/*current gpu device id*/
	int device;

	/*lock virtio*/
	spinlock_t io_lock;
};

/* This is the very early arch-specified put chars function. */
static int (*early_put_chars)(u32, const char *, int);

static struct port *find_port_by_devt_in_portdev(struct ports_device *portdev,
						 dev_t dev)
{
	struct port *port;
	unsigned long flags;

	spin_lock_irqsave(&portdev->ports_lock, flags);
	list_for_each_entry(port, &portdev->ports, list) {
		if (port->cdev->dev == dev) {
			kref_get(&port->kref);
			goto out;
		}
	}
	port = NULL;
out:
	spin_unlock_irqrestore(&portdev->ports_lock, flags);

	return port;
}

static struct port *find_port_by_devt(dev_t dev)
{
	struct ports_device *portdev;
	struct port *port;
	unsigned long flags;

	spin_lock_irqsave(&pdrvdata_lock, flags);
	list_for_each_entry(portdev, &pdrvdata.portdevs, list) {
		port = find_port_by_devt_in_portdev(portdev, dev);
		if (port)
			goto out;
	}
	port = NULL;
out:
	spin_unlock_irqrestore(&pdrvdata_lock, flags);
	return port;
}

static struct port *find_port_by_id(struct ports_device *portdev, u32 id)
{
	struct port *port;
	unsigned long flags;

	spin_lock_irqsave(&portdev->ports_lock, flags);
	list_for_each_entry(port, &portdev->ports, list)
		if (port->id == id)
			goto out;
	port = NULL;
out:
	spin_unlock_irqrestore(&portdev->ports_lock, flags);

	return port;
}

static struct port *find_port_by_vq(struct ports_device *portdev,
				    struct virtqueue *vq)
{
	struct port *port;
	unsigned long flags;

	spin_lock_irqsave(&portdev->ports_lock, flags);
	list_for_each_entry(port, &portdev->ports, list)
		if (port->in_vq == vq || port->out_vq == vq)
			goto out;
	port = NULL;
out:
	spin_unlock_irqrestore(&portdev->ports_lock, flags);
	return port;
}

static bool is_console_port(struct port *port)
{
	if (port->cons.hvc)
		return true;
	return false;
}

static bool is_rproc_serial(const struct virtio_device *vdev)
{
	return is_rproc_enabled && vdev->id.device == VIRTIO_ID_RPROC_SERIAL;
}

static inline bool use_multiport(struct ports_device *portdev)
{
	/*
	 * This condition can be true when put_chars is called from
	 * early_init
	 */
	if (!portdev->vdev)
		return false;
	return __virtio_test_bit(portdev->vdev, VIRTIO_CONSOLE_F_MULTIPORT);
}

static DEFINE_SPINLOCK(dma_bufs_lock);
static LIST_HEAD(pending_free_dma_bufs);

static void free_buf(struct port_buffer *buf, bool can_sleep)
{
	unsigned int i;

	for (i = 0; i < buf->sgpages; i++) {
		struct page *page = sg_page(&buf->sg[i]);
		if (!page)
			break;
		put_page(page);
	}

	if (!buf->dev) {
		kfree(buf->buf);
	} else if (is_rproc_enabled) {
		unsigned long flags;

		/* dma_free_coherent requires interrupts to be enabled. */
		if (!can_sleep) {
			/* queue up dma-buffers to be freed later */
			spin_lock_irqsave(&dma_bufs_lock, flags);
			list_add_tail(&buf->list, &pending_free_dma_bufs);
			spin_unlock_irqrestore(&dma_bufs_lock, flags);
			return;
		}
		dma_free_coherent(buf->dev, buf->size, buf->buf, buf->dma);

		/* Release device refcnt and allow it to be freed */
		put_device(buf->dev);
	}

	kfree(buf);
}

static void reclaim_dma_bufs(void)
{
	unsigned long flags;
	struct port_buffer *buf, *tmp;
	LIST_HEAD(tmp_list);

	if (list_empty(&pending_free_dma_bufs))
		return;
	
	/* Create a copy of the pending_free_dma_bufs while holding the lock */
	spin_lock_irqsave(&dma_bufs_lock, flags);
	list_cut_position(&tmp_list, &pending_free_dma_bufs,
			  pending_free_dma_bufs.prev);
	spin_unlock_irqrestore(&dma_bufs_lock, flags);

	/* Release the dma buffers, without irqs enabled */
	list_for_each_entry_safe(buf, tmp, &tmp_list, list) {
		list_del(&buf->list);
		free_buf(buf, true);
	}
}

static struct port_buffer *alloc_buf(struct virtqueue *vq, size_t buf_size,
				     int pages)
{
	struct port_buffer *buf;

	reclaim_dma_bufs();

	/*
	 * Allocate buffer and the sg list. The sg list array is allocated
	 * directly after the port_buffer struct.
	 */
	buf = kmalloc(sizeof(*buf) + sizeof(struct scatterlist) * pages,
		      GFP_KERNEL);
	if (!buf)
		goto fail;

	buf->sgpages = pages;
	if (pages > 0) {
		buf->dev = NULL;
		buf->buf = NULL;
		return buf;
	}

	if (is_rproc_serial(vq->vdev)) {
		/*
		 * Allocate DMA memory from ancestor. When a virtio
		 * device is created by remoteproc, the DMA memory is
		 * associated with the grandparent device:
		 * vdev => rproc => platform-dev.
		 */
		if (!vq->vdev->dev.parent || !vq->vdev->dev.parent->parent)
			goto free_buf;
		buf->dev = vq->vdev->dev.parent->parent;

		/* Increase device refcnt to avoid freeing it */
		get_device(buf->dev);
		buf->buf = dma_alloc_coherent(buf->dev, buf_size, &buf->dma,
					      GFP_KERNEL);
	} else {
		buf->dev = NULL;
		buf->buf = kmalloc(buf_size, GFP_KERNEL);
	}

	if (!buf->buf)
		goto free_buf;
	buf->len = 0;
	buf->offset = 0;
	buf->size = buf_size;
	return buf;

free_buf:
	kfree(buf);
fail:
	return NULL;
}

/* Callers should take appropriate locks */
static struct port_buffer *get_inbuf(struct port *port)
{
	struct port_buffer *buf;
	unsigned int len;

	if (port->inbuf)
		return port->inbuf;

	buf = virtqueue_get_buf(port->in_vq, &len);
	if (buf) {
		buf->len = len;
		buf->offset = 0;
		port->stats.bytes_received += len;
	}
	return buf;
}

/*
 * Create a scatter-gather list representing our input buffer and put
 * it in the queue.
 *
 * Callers should take appropriate locks.
 */
static int add_inbuf(struct virtqueue *vq, struct port_buffer *buf)
{
	struct scatterlist sg[1];
	int ret;

	sg_init_one(sg, buf->buf, buf->size);

	ret = virtqueue_add_inbuf(vq, sg, 1, buf, GFP_ATOMIC);
	virtqueue_kick(vq);
	if (!ret)
		ret = vq->num_free;
	return ret;
}

/* Discard any unread data this port has. Callers lockers. */
static void discard_port_data(struct port *port)
{
	struct port_buffer *buf;
	unsigned int err;

	if (!port->portdev) {
		/* Device has been unplugged.  vqs are already gone. */
		return;
	}
	buf = get_inbuf(port);
	
	err = 0;
	while (buf) {
		port->stats.bytes_discarded += buf->len - buf->offset;
		if (add_inbuf(port->in_vq, buf) < 0) {
			err++;
			free_buf(buf, false);
		}
		port->inbuf = NULL;
		buf = get_inbuf(port);
	}
	if (err)
		dev_warn(port->dev, "Errors adding %d buffers back to vq\n",
			 err);
}

static bool port_has_data(struct port *port)
{
	unsigned long flags;
	bool ret;

	ret = false;
	spin_lock_irqsave(&port->inbuf_lock, flags);
	port->inbuf = get_inbuf(port);
	if (port->inbuf)
		ret = true;

	spin_unlock_irqrestore(&port->inbuf_lock, flags);
	return ret;
}

static ssize_t __send_control_msg(struct ports_device *portdev, u32 port_id,
				  unsigned int event, unsigned int value)
{
	struct scatterlist sg[1];
	struct virtqueue *vq;
	unsigned int len;

	if (!use_multiport(portdev))
		return 0;

	vq = portdev->c_ovq;

	spin_lock(&portdev->c_ovq_lock);

	portdev->cpkt.id = cpu_to_virtio32(portdev->vdev, port_id);
	portdev->cpkt.event = cpu_to_virtio16(portdev->vdev, event);
	portdev->cpkt.value = cpu_to_virtio16(portdev->vdev, value);

	sg_init_one(sg, &portdev->cpkt, sizeof(struct virtio_console_control));

	if (virtqueue_add_outbuf(vq, sg, 1, &portdev->cpkt, GFP_ATOMIC) == 0) {
		virtqueue_kick(vq);
		while (!virtqueue_get_buf(vq, &len)
			&& !virtqueue_is_broken(vq))
			cpu_relax();
	}

	spin_unlock(&portdev->c_ovq_lock);
	return 0;
}

static ssize_t send_control_msg(struct port *port, unsigned int event,
				unsigned int value)
{
	/* Did the port get unplugged before userspace closed it? */
	if (port->portdev)
		return __send_control_msg(port->portdev, port->id, event, value);
	return 0;
}


/* Callers must take the port->outvq_lock */
static void reclaim_consumed_buffers(struct port *port)
{
	struct port_buffer *buf;
	unsigned int len;

	if (!port->portdev) {
		/* Device has been unplugged.  vqs are already gone. */
		return;
	}
	
	while ((buf = virtqueue_get_buf(port->out_vq, &len))) {
		free_buf(buf, false);
		port->outvq_full = false;
	}
}

static ssize_t __send_to_port(struct port *port, struct scatterlist *sg,
			      int nents, size_t in_count,
			      void *data, bool nonblock)
{
	struct virtqueue *out_vq;
	int err;
	unsigned long flags;
	unsigned int len;

	out_vq = port->out_vq;

	spin_lock_irqsave(&port->outvq_lock, flags);

	reclaim_consumed_buffers(port);

	err = virtqueue_add_outbuf(out_vq, sg, nents, data, GFP_ATOMIC);

	/* Tell Host to go! */
	virtqueue_kick(out_vq);

	if (err) {
		in_count = 0;
		goto done;
	}

	if (out_vq->num_free == 0)
		port->outvq_full = true;

	if (nonblock)
		goto done;

	/*
	 * Wait till the host acknowledges it pushed out the data we
	 * sent.  This is done for data from the hvc_console; the tty
	 * operations are performed with spinlocks held so we can't
	 * sleep here.  An alternative would be to copy the data to a
	 * buffer and relax the spinning requirement.  The downside is
	 * we need to kmalloc a GFP_ATOMIC buffer each time the
	 * console driver writes something out.
	 */
	while (!virtqueue_get_buf(out_vq, &len)
		&& !virtqueue_is_broken(out_vq))
		cpu_relax();
done:
	spin_unlock_irqrestore(&port->outvq_lock, flags);

	port->stats.bytes_sent += in_count;
	/*
	 * We're expected to return the amount of data we wrote -- all
	 * of it
	 */
	return in_count;
}

/*
 * Give out the data that's requested from the buffer that we have
 * queued up.
 */
static ssize_t fill_readbuf(struct port *port, char __user *out_buf,
			    size_t out_count, bool to_user)
{
	struct port_buffer *buf;
	unsigned long flags;

	if (!out_count || !port_has_data(port))
		return 0;

	buf = port->inbuf;
	out_count = min(out_count, buf->len - buf->offset);

	if (to_user) {
		ssize_t ret;

		ret = copy_to_user(out_buf, buf->buf + buf->offset, out_count);
		if (ret)
			return -EFAULT;
	} else {
		memcpy((__force char *)out_buf, buf->buf + buf->offset,
		       out_count);
	}

	buf->offset += out_count;

	if (buf->offset == buf->len) {
		/*
		 * We're done using all the data in this buffer.
		 * Re-queue so that the Host can send us more data.
		 */
		spin_lock_irqsave(&port->inbuf_lock, flags);
		port->inbuf = NULL;

		if (add_inbuf(port->in_vq, buf) < 0)
			dev_warn(port->dev, "failed add_buf\n");

		spin_unlock_irqrestore(&port->inbuf_lock, flags);
	}
	/* Return the number of bytes actually copied */
	return out_count;
}

/* The condition that must be true for polling to end */
static bool will_read_block(struct port *port)
{
	if (!port->guest_connected) {
		/* Port got hot-unplugged. Let's exit. */
		return false;
	}
	return !port_has_data(port) && port->host_connected;
}

static bool will_write_block(struct port *port)
{
	bool ret;

	if (!port->guest_connected) {
		/* Port got hot-unplugged. Let's exit. */
		return false;
	}
	if (!port->host_connected)
		return true;

	spin_lock_irq(&port->outvq_lock);
	/*
	 * Check if the Host has consumed any buffers since we last
	 * sent data (this is only applicable for nonblocking ports).
	 */
	reclaim_consumed_buffers(port);
	ret = port->outvq_full;
	spin_unlock_irq(&port->outvq_lock);

	return ret;
}

static ssize_t port_fops_read(struct file *filp, char __user *ubuf,
			      size_t count, loff_t *offp)
{
	struct port *port;
	ssize_t ret;

	port = filp->private_data;

	/* Port is hot-unplugged. */
	if (!port->guest_connected)
		return -ENODEV;

	if (!port_has_data(port)) {
		/*
		 * If nothing's connected on the host just return 0 in
		 * case of list_empty; this tells the userspace app
		 * that there's no connection
		 */
		if (!port->host_connected)
			return 0;
		if (filp->f_flags & O_NONBLOCK)
			return -EAGAIN;

		ret = wait_event_freezable(port->waitqueue,
					   !will_read_block(port));
		if (ret < 0)
			return ret;
	}
	/* Port got hot-unplugged while we were waiting above. */
	if (!port->guest_connected)
		return -ENODEV;
	/*
	 * We could've received a disconnection message while we were
	 * waiting for more data.
	 *
	 * This check is not clubbed in the if() statement above as we
	 * might receive some data as well as the host could get
	 * disconnected after we got woken up from our wait.  So we
	 * really want to give off whatever data we have and only then
	 * check for host_connected.
	 */
	if (!port_has_data(port) && !port->host_connected)
		return 0;

	return fill_readbuf(port, ubuf, count, true);
}

static int wait_port_writable(struct port *port, bool nonblock)
{
	int ret;

	if (will_write_block(port)) {
		if (nonblock)
			return -EAGAIN;

		ret = wait_event_freezable(port->waitqueue,
					   !will_write_block(port));
		if (ret < 0)
			return ret;
	}
	/* Port got hot-unplugged. */
	if (!port->guest_connected)
		return -ENODEV;

	return 0;
}

static ssize_t port_fops_write(struct file *filp, const char __user *ubuf,
			       size_t count, loff_t *offp)
{
	struct port *port;
	struct port_buffer *buf;
	ssize_t ret;
	bool nonblock;
	struct scatterlist sg[1];
	func();
	/* Userspace could be out to fool us */
	if (!count)
		return 0;

	port = filp->private_data;

	nonblock = filp->f_flags & O_NONBLOCK;

	ret = wait_port_writable(port, nonblock);
	if (ret < 0)
		return ret;

	count = min((size_t)(32 * 1024), count);

	buf = alloc_buf(port->out_vq, count, 0);
	if (!buf)
		return -ENOMEM;

	ret = copy_from_user(buf->buf, ubuf, count);
	if (ret) {
		ret = -EFAULT;
		goto free_buf;
	}

	/*
	 * We now ask send_buf() to not spin for generic ports -- we
	 * can re-use the same code path that non-blocking file
	 * descriptors take for blocking file descriptors since the
	 * wait is already done and we're certain the write will go
	 * through to the host.
	 */
	nonblock = true;
	sg_init_one(sg, buf->buf, count);
	ret = __send_to_port(port, sg, 1, count, buf, nonblock);

	if (nonblock && ret > 0)
		goto out;

free_buf:
	free_buf(buf, true);
out:
	return ret;
}

struct sg_list {
	unsigned int n;
	unsigned int size;
	size_t len;
	struct scatterlist *sg;
};

static int pipe_to_sg(struct pipe_inode_info *pipe, struct pipe_buffer *buf,
			struct splice_desc *sd)
{
	struct sg_list *sgl = sd->u.data;
	unsigned int offset, len;

	if (sgl->n == sgl->size)
		return 0;

	/* Try lock this page */
	if (pipe_buf_steal(pipe, buf) == 0) {
		/* Get reference and unlock page for moving */
		get_page(buf->page);
		unlock_page(buf->page);

		len = min(buf->len, sd->len);
		sg_set_page(&(sgl->sg[sgl->n]), buf->page, len, buf->offset);
	} else {
		/* Failback to copying a page */
		struct page *page = alloc_page(GFP_KERNEL);
		char *src;

		if (!page)
			return -ENOMEM;

		offset = sd->pos & ~PAGE_MASK;

		len = sd->len;
		if (len + offset > PAGE_SIZE)
			len = PAGE_SIZE - offset;

		src = kmap_atomic(buf->page);
		memcpy(page_address(page) + offset, src + buf->offset, len);
		kunmap_atomic(src);

		sg_set_page(&(sgl->sg[sgl->n]), page, len, offset);
	}
	sgl->n++;
	sgl->len += len;

	return len;
}

/* Faster zero-copy write by splicing */
static ssize_t port_fops_splice_write(struct pipe_inode_info *pipe,
				      struct file *filp, loff_t *ppos,
				      size_t len, unsigned int flags)
{
	struct port *port = filp->private_data;
	struct sg_list sgl;
	ssize_t ret;
	struct port_buffer *buf;
	struct splice_desc sd = {
		.total_len = len,
		.flags = flags,
		.pos = *ppos,
		.u.data = &sgl,
	};

	/*
	 * Rproc_serial does not yet support splice. To support splice
	 * pipe_to_sg() must allocate dma-buffers and copy content from
	 * regular pages to dma pages. And alloc_buf and free_buf must
	 * support allocating and freeing such a list of dma-buffers.
	 */
	if (is_rproc_serial(port->out_vq->vdev))
		return -EINVAL;

	/*
	 * pipe->nrbufs == 0 means there are no data to transfer,
	 * so this returns just 0 for no data.
	 */
	pipe_lock(pipe);
	if (!pipe->nrbufs) {
		ret = 0;
		goto error_out;
	}

	ret = wait_port_writable(port, filp->f_flags & O_NONBLOCK);
	if (ret < 0)
		goto error_out;

	buf = alloc_buf(port->out_vq, 0, pipe->nrbufs);
	if (!buf) {
		ret = -ENOMEM;
		goto error_out;
	}

	sgl.n = 0;
	sgl.len = 0;
	sgl.size = pipe->nrbufs;
	sgl.sg = buf->sg;
	sg_init_table(sgl.sg, sgl.size);
	ret = __splice_from_pipe(pipe, &sd, pipe_to_sg);
	pipe_unlock(pipe);
	if (likely(ret > 0))
		ret = __send_to_port(port, buf->sg, sgl.n, sgl.len, buf, true);

	if (unlikely(ret <= 0))
		free_buf(buf, true);
	return ret;

error_out:
	pipe_unlock(pipe);
	return ret;
}

static unsigned int port_fops_poll(struct file *filp, poll_table *wait)
{
	struct port *port;
	unsigned int ret;

	port = filp->private_data;
	poll_wait(filp, &port->waitqueue, wait);

	if (!port->guest_connected) {
		/* Port got unplugged */
		return POLLHUP;
	}
	ret = 0;
	if (!will_read_block(port))
		ret |= POLLIN | POLLRDNORM;
	if (!will_write_block(port))
		ret |= POLLOUT;
	if (!port->host_connected)
		ret |= POLLHUP;

	return ret;
}

static void remove_port(struct kref *kref);

static int port_fops_release(struct inode *inode, struct file *filp)
{
	struct port *port;
	struct vgpu_device *vgpu;
	unsigned long flags;

	func();

	port = filp->private_data;
	/*cleaning GPU flags */
	spin_lock_irqsave(&port->portdev->vgpus_lock, flags);
	list_for_each_entry(vgpu, &port->portdev->vgpus, list) {
		vgpu->initialized = 0;
		vgpu->flags = 0;
	}
	spin_unlock_irqrestore(&port->portdev->vgpus_lock, flags);

	/* Notify host of port being closed */
	send_control_msg(port, VIRTIO_CONSOLE_PORT_OPEN, 0);

	spin_lock_irq(&port->inbuf_lock);
	port->guest_connected = false;

	discard_port_data(port);

	spin_unlock_irq(&port->inbuf_lock);

	spin_lock_irq(&port->outvq_lock);
	reclaim_consumed_buffers(port);
	spin_unlock_irq(&port->outvq_lock);

	reclaim_dma_bufs();
	/*
	 * Locks aren't necessary here as a port can't be opened after
	 * unplug, and if a port isn't unplugged, a kref would already
	 * exist for the port.  Plus, taking ports_lock here would
	 * create a dependency on other locks taken by functions
	 * inside remove_port if we're the last holder of the port,
	 * creating many problems.
	 */
	kref_put(&port->kref, remove_port);

	return 0;
}

static int port_fops_open(struct inode *inode, struct file *filp)
{
	struct cdev *cdev = inode->i_cdev;
	struct port *port;
	int ret;

	func();

	/* We get the port with a kref here */
	port = find_port_by_devt(cdev->dev);
	if (!port) {
		/* Port was unplugged before we could proceed */
		return -ENXIO;
	}
	filp->private_data = port;

	/*Whenever user opens the vgpu device, Init device id and device memory list */
	gldebug("filp->private_data port->id=%d\n", port->id);
	port->device = 0;
	INIT_LIST_HEAD(&port->device_mem_list);
	INIT_LIST_HEAD(&port->guest_mem_list);
	/*
	 * Don't allow opening of console port devices -- that's done
	 * via /dev/hvc
	 */
	if (is_console_port(port)) {
		ret = -ENXIO;
		goto out;
	}

	/* Allow only ||one process|| to open a particular port at a time */
	spin_lock_irq(&port->inbuf_lock);
	if (port->guest_connected) {
		spin_unlock_irq(&port->inbuf_lock);
		ret = -EBUSY;
		goto out;
	}

	port->guest_connected = true;
	spin_unlock_irq(&port->inbuf_lock);

	spin_lock_irq(&port->outvq_lock);
	/*
	 * There might be a chance that we missed reclaiming a few
	 * buffers in the window of the port getting previously closed
	 * and opening now.
	 */
	reclaim_consumed_buffers(port);
	spin_unlock_irq(&port->outvq_lock);

	nonseekable_open(inode, filp);

	/* Notify host of port being opened */
	send_control_msg(filp->private_data, VIRTIO_CONSOLE_PORT_OPEN, 1);

	return 0;
out:
	kref_put(&port->kref, remove_port);
	return ret;
}

static int port_fops_fasync(int fd, struct file *filp, int mode)
{
	struct port *port;

	port = filp->private_data;
	return fasync_helper(fd, filp, mode, &port->async_queue);
}

static struct vgpu_device *find_gpu_by_device(struct ports_device *portdev, u32 device)
{
	struct vgpu_device *gpu;
	unsigned long flags;

	spin_lock_irqsave(&portdev->vgpus_lock, flags);
	list_for_each_entry(gpu, &portdev->vgpus, list)
		if (gpu->id == device)
			goto out;
	gpu = NULL;
out:
	spin_unlock_irqrestore(&portdev->vgpus_lock, flags);

	return gpu;
}

static struct virtio_uvm_page *find_page_by_addr(unsigned long addr, 
													struct port *port)
{
	struct virtio_uvm_page *page;
	list_for_each_entry(page, &port->page, list) {
		if(page->uvm_start <= addr && addr < page->uvm_end)
			return page;
	}
	return NULL;
}

static struct virtio_uvm_page *find_pages_by_addr(unsigned long addr, 
													struct list_head *head)
{
	struct virtio_uvm_page *page;
	list_for_each_entry(page, head, list) {
		if(page->uvm_start <= addr && addr < page->uvm_end)
			return page;
	}
	return NULL;
}

#define PAGE_UP(addr)  		(((addr)+((PAGE_SIZE)-1))&(~((PAGE_SIZE)-1)))
#define PAGE_DOWN(addr) 	((addr)&(~((PAGE_SIZE)-1)))
#define page_nr(addr, size) ((PAGE_UP((addr)+(size))>>PAGE_SHIFT)-(PAGE_DOWN(addr)>>PAGE_SHIFT))

static struct page ** uaddr_to_pages(const unsigned long __user buf, size_t size, int write_flag)
{
    struct  page **page_list;
    unsigned int user_pages_flags = 0;
    unsigned long nr_pages = 0, addr, next;
    int i, nr;

    func();
    // if the attribute of mmapped file is O_RDONLY, then FOLL_WRITE will be excluded
    // So, I decide to leave out the user_pages_flags
    if (write_flag)
    	user_pages_flags |= FOLL_WRITE;
    gldebug("buf 0x%lx, size 0x%lx\n", (unsigned long)buf, size);
    nr_pages = page_nr(buf, size);
    page_list = kvmalloc_array(nr_pages, sizeof(struct page *), GFP_KERNEL);
    if(!page_list) {
    	pr_err("malloc page list error.\n"); 
    	return NULL;
    }
    gldebug("page num 0x%lx\n", nr_pages);
    i=0;
    nr = CHUNK_SIZE>>PAGE_SHIFT;
    gldebug("end addr %lx\n", buf+size);
    for(addr=buf; addr < buf+size; addr=next) {
    	next = addr + (nr << PAGE_SHIFT);
    	// gldebug("next %lx, pagealign %lx\n", next, PAGE_ALIGN(next));
    	if(next >= buf+size) {
    		next = buf + size;
    		nr = (PAGE_ALIGN(next)-(PAGE_DOWN(addr)))>>PAGE_SHIFT;
    	}
    	// gldebug("page nr 0x%x\n", nr);
	    nr = get_user_pages_fast(addr, nr, user_pages_flags, page_list+i);
	    if(nr <= 0) {
	    	pr_err("get_user_pages_fast return error ret %d\n", nr);
	    	break;
	    }
	    i+=nr;
    }
    // gldebug("page num = 0x%lx, get_user_pages_fast return %x\n", nr_pages, i);
    if(i != nr_pages) {
    	pr_err("[ERROR] #page pinned 0x%x != requested 0x%lx\n", i, nr_pages);
    }
    
    for(i=0; i<nr_pages; i++) {
		if(!page_list[i])
			break;
		// gldebug("page %d pfn %lx\n", i, page_to_pfn(page_list[i]));
		put_page(page_list[i]);
    }
    // !!dont forget kvfree(page_list);
    return page_list;
}

void send_sgs_to_virtio(struct port *port, struct scatterlist *sg[], int out_nr, int in_nr)
{
	unsigned int err = 0;
	bool nonblock=false;
	struct virtqueue *out_vq;
	unsigned long flags;
	unsigned int len;

	out_vq = port->out_vq;
	spin_lock_irqsave(&port->outvq_lock, flags);
	reclaim_consumed_buffers(port);
	err = virtqueue_add_sgs(out_vq, sg, out_nr, in_nr, (void*)sg, GFP_ATOMIC);

	/* Tell Host to go! */
	virtqueue_kick(out_vq);

	if (err) {
		goto done;
	}

	if (out_vq->num_free == 0)
		port->outvq_full = true;

	if (nonblock)
		goto done;

	while (!virtqueue_get_buf(out_vq, &len)
		&& !virtqueue_is_broken(out_vq))
		cpu_relax();
done:
	spin_unlock_irqrestore(&port->outvq_lock, flags);
	gldebug("Finish sending data\n");
}

int wait_for_inbuf(struct port *port, char *arg, int count)
{
	int err = 0;
	long recv = 0;
	//now read data from host
	/* Port is hot-unplugged. */
	if (!port->guest_connected){
		return -ENODEV;
	}

	if (!port_has_data(port)) {
		/*
		 * If nothing's connected on the host just return 0 in
		 * case of list_empty; this tells the userspace app
		 * that there's no connection
		 */
		if (!port->host_connected){
			return -ENODEV;
		}
		err = wait_event_freezable(port->waitqueue, !will_read_block(port));
		if (err < 0){
			return -ENODEV;
		}
	}
	// Port got hot-unplugged while we were waiting above. 
	if (!port->guest_connected){
		return -ENODEV;
	}
	/*
	 * We could've received a disconnection message while we were
	 * waiting for more data.
	 *
	 * This check is not clubbed in the if() statement above as we
	 * might receive some data as well as the host could get
	 * disconnected after we got woken up from our wait.  So we
	 * really want to give off whatever data we have and only then
	 * check for host_connected.
	 */
	if (!port_has_data(port) && !port->host_connected){
		return -ENODEV;
	}
	recv = fill_readbuf(port, arg, count, true);
	gldebug("receiving %zu data\n", recv);
	return 0;
}

int send_to_virtio(struct port *port, void *payload, size_t count)
{
	struct scatterlist sg[1];
	unsigned int err = 0;
	long recv = 0;
	void *data;
	// unsigned long flags;

	data = kmemdup(payload, count, GFP_ATOMIC);
	if(!data) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", count);	
		return -ENOMEM;
	}
	sg_init_one(sg, data, count);
	err = __send_to_port(port, sg, 1, count, data, false);
	if (err <= 0) {
		pr_err("[ERROR] send to port error. No data return.\n");
		return -EINVAL;
	}
	kfree(data);
	gldebug("Finish sending data\n");

	//now read data from host
	/* Port is hot-unplugged. */
	if (!port->guest_connected){
		return -ENODEV;
	}

	if (!port_has_data(port)) {
		/*
		 * If nothing's connected on the host just return 0 in
		 * case of list_empty; this tells the userspace app
		 * that there's no connection
		 */
		if (!port->host_connected){
			return -ENODEV;
		}
		err = wait_event_freezable(port->waitqueue, !will_read_block(port));
		if (err < 0){
			return -ENODEV;
		}
	}
	// Port got hot-unplugged while we were waiting above. 
	if (!port->guest_connected){
		return -ENODEV;
	}
	/*
	 * We could've received a disconnection message while we were
	 * waiting for more data.
	 *
	 * This check is not clubbed in the if() statement above as we
	 * might receive some data as well as the host could get
	 * disconnected after we got woken up from our wait.  So we
	 * really want to give off whatever data we have and only then
	 * check for host_connected.
	 */
	if (!port_has_data(port) && !port->host_connected){
		return -ENODEV;
	}
	// recv = fill_readbuf(port, out , count, false);
	recv = fill_readbuf(port, payload, count, false);
	gldebug("receiving %zu data\n", recv);
	return 0;
}

static int cuda_primarycontext(VirtIOArg __user *arg, struct port *port)
{
	void *gva;
	VirtIOArg *payload;
	uint32_t src_size;
	struct scatterlist *sgs[2], arg_sg, gva_sg;
	int num_out = 0;
	struct vgpu_device *vgpu;

	func();
	vgpu = find_gpu_by_device(port->portdev, port->device);
	if (!vgpu) {
		pr_err("Failed to find properties of device id %d.\n", port->device);
		// cudaErrorInvalidDevice                =     10,
		// put_user(10, &arg->cmd);
		return -ENODEV;
	}
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	src_size = payload->srcSize;
	gva = memdup_user((const void __user *)payload->src, (size_t)src_size);
	if(!gva) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", src_size);	
		kfree(payload);
		return -ENOMEM;
	}
	gldebug("memdup 0x%x size\n", src_size);
	sg_init_one(&arg_sg, payload, sizeof(*payload));
	sgs[num_out++] = &arg_sg;
	sg_init_one(&gva_sg, gva, src_size);
	sgs[num_out++] = &gva_sg;
	
	send_sgs_to_virtio(port, sgs, num_out, 0);
	// ret = wait_for_inbuf(port, (char *)arg, ARG_SIZE);
	gldebug("[+] Now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	vgpu->initialized = 1;
	copy_to_user(arg, payload, ARG_SIZE);
	kfree(gva);
	kfree(payload);
	return 0;
}

static int send_single_payload(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	struct scatterlist *sgs[1], arg_sg;
	int num_out = 0, num_in = 0;

	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	gldebug("src=0x%llx, srcSize=0x%x, dst=0x%llx, dstSize=0x%x, "
			"flag=%llu, param=0x%llx\n",
			payload->src, payload->srcSize, payload->dst, payload->dstSize, 
			payload->flag, payload->param);
	sg_init_one(&arg_sg, payload, sizeof(*payload));
	sgs[num_out++] = &arg_sg;

#ifdef VIRTIO_LOCK
	spin_lock(&port->io_lock);
#endif
	send_sgs_to_virtio(port, sgs, num_out, num_in);
#ifdef VIRTIO_LOCK
	spin_unlock(&port->io_lock);
#endif

	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	copy_to_user(arg, payload, ARG_SIZE);
	kfree(payload);
	return 0;
}

int cuda_register_fatbinary(VirtIOArg __user *arg, struct port *port)
{
	return 0;
}

int cuda_unregister_fatbinary(VirtIOArg __user *arg, struct port *port)
{
	return 0;
}

int cuda_register_function(VirtIOArg __user *arg, struct port *port)
{
	return 0;
}

int cuda_register_var(VirtIOArg __user *arg, struct port *port)
{
	return 0;
}

/*Fix me*/
int cuda_launch(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	void *para;
	int ret;
	uint32_t para_size;

	func();
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	para_size = payload->srcSize;
	para = memdup_user((const void __user *)payload->src, (size_t)para_size);
	if(!para) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", para_size);	
		return -ENOMEM;
	}

	payload->src = (uint64_t)virt_to_phys(para);

	#ifdef VIRTIO_LOCK
	spin_lock(&port->io_lock);
	#endif

	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	#ifdef VIRTIO_LOCK
	spin_unlock(&port->io_lock);
	#endif

	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	kfree(para);
	kfree(payload);
	return ret;
}

static int cuda_launch_kernel(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	void *para;
	uint32_t para_buf_size;
	struct scatterlist *sgs[2], arg_sg, param_sg;
	int num_out=0, num_in=0;

	func();
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	para_buf_size = payload->srcSize;
	para = memdup_user((const void __user *)payload->src, (size_t)para_buf_size);
	if(!para) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", para_buf_size);
		kfree(payload);
		return -ENOMEM;
	}

	sg_init_one(&arg_sg, payload, sizeof(*payload));
	sgs[num_out++] = &arg_sg;
	sg_init_one(&param_sg, para, para_buf_size);
	sgs[num_out++] = &param_sg;

#ifdef VIRTIO_LOCK
	spin_lock(&port->io_lock);
#endif
	send_sgs_to_virtio(port, sgs, num_out, num_in);
#ifdef VIRTIO_LOCK
	spin_unlock(&port->io_lock);
#endif

	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	copy_to_user(arg, payload, ARG_SIZE);
	kfree(para);
	kfree(payload);
	return 0;
}


static int cuda_malloc(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

static int cuda_free(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

int cuda_malloc_host(VirtIOArg __user *arg, struct port *port)
{
	return 0;
}

int cuda_free_host(VirtIOArg __user *arg, struct port *port)
{
	return 0;
}

static int cuda_memcpy_to_symbol(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	void *va = NULL;
	struct scatterlist *sgs[2], arg_sg, src_sg;
	int num_out = 0, num_in = 0;

	func();
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	va = memdup_user((void*)payload->src, payload->srcSize);
	if(!va) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", payload->srcSize);
		kfree(payload);
		return -ENOMEM;
	}
	gldebug("src=0x%llx, srcSize=0x%x, dst=0x%llx, dstSize=0x%x, "
			"flag=%llu, param=0x%llx\n",
			payload->src, payload->srcSize, payload->dst, payload->dstSize,
			payload->flag, payload->param);
	sg_init_one(&arg_sg, payload, sizeof(*payload));
	sgs[num_out++] = &arg_sg;
	sg_init_one(&src_sg, va, payload->srcSize);
	sgs[num_out++] = &src_sg;

#ifdef VIRTIO_LOCK
	spin_lock(&port->io_lock);
#endif
	send_sgs_to_virtio(port, sgs, num_out, num_in);
#ifdef VIRTIO_LOCK
	spin_unlock(&port->io_lock);
#endif

	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	copy_to_user(arg, payload, ARG_SIZE);
	kfree(va);
	kfree(payload);
	return 0;
}

static int cuda_memcpy_from_symbol(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	void *va = NULL;
	uint32_t size;
	struct scatterlist *sgs[2], arg_sg, host_sg;
	int num_out = 0, num_in = 0;

	func();
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	size = payload->dstSize;
	// cudaMemcpyDeviceToHost
	va = kmalloc(size, GFP_KERNEL);
	if(!va) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", size);
		kfree(payload);
		return -ENOMEM;
	}
	sg_init_one(&arg_sg, payload, sizeof(*payload));
	sgs[num_out++] = &arg_sg;
	sg_init_one(&host_sg, va, size);
	sgs[num_out + num_in++] = &host_sg;

#ifdef VIRTIO_LOCK
	spin_lock(&port->io_lock);
#endif
	send_sgs_to_virtio(port, sgs, num_out, num_in);
#ifdef VIRTIO_LOCK
	spin_unlock(&port->io_lock);
#endif

	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	copy_to_user(arg, payload, ARG_SIZE);
	// pay attention, do not use user pointer in kernel
	if(copy_to_user((void __user *)payload->dst, va, size)) {
		pr_err("[ERROR] Failed to copy to user \n");
	}
	kfree(va);
	kfree(payload);
	return 0;
}

static int cuda_host_register(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	uint32_t src_size;
	struct scatterlist *sgs[2], arg_sg;
	int num_out = 0, num_in = 0;
	struct page ** page_list;
	int nr_pages=0;
	struct virtio_uvm_page *pages= NULL;
	struct sg_table *st;
	int ret = 0;
	
	func();
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	src_size = payload->srcSize;
	nr_pages = page_nr(payload->src, src_size);
	gldebug("nr page %x\n", nr_pages);
	pages = kmalloc(sizeof(struct virtio_uvm_page),GFP_KERNEL);
	list_add_tail(&pages->list, &port->page);
	pages->uvm_start = payload->src;
	pages->uvm_end 	= payload->src + src_size;

	st = kmalloc(sizeof(*st), GFP_KERNEL);
	if(!st) {
		kfree(payload);
		return -ENOMEM;
	}
	pages->st = st;
	page_list = uaddr_to_pages(payload->src, src_size, 0);
	if(!page_list) {
		// cudaErrorMemoryAllocation             =      2,
		// put_user(2, &arg->cmd);
		return -ENOMEM;
	}
	ret = sg_alloc_table_from_pages(st, page_list, nr_pages, 
				offset_in_page(payload->src), src_size, GFP_KERNEL);
	if(ret < 0) {
		pr_err("Failed to allocated sg table, ret %d\n", ret);
		// put_user(2, &arg->cmd);
		return -ENOMEM;
	}
	kvfree(page_list);
	gldebug("sg nents %d\n", st->nents);
	if (st->nents > VIRTIO_INDIRECT_NUM_MAX) {
		pr_err("Pages num exceed %d\n", VIRTIO_INDIRECT_NUM_MAX);
		// cudaErrorMemoryAllocation             =      2,
		// put_user(2, &arg->cmd);
		kfree(payload);
		return -ENOMEM;
	}
	
	sg_init_one(&arg_sg, payload, sizeof(*payload));
	sgs[num_out++] = &arg_sg;
	sgs[num_out++] = st->sgl;
#ifdef VIRTIO_LOCK
	spin_lock(&port->io_lock);
#endif
	send_sgs_to_virtio(port, sgs, num_out, num_in);
#ifdef VIRTIO_LOCK
	spin_unlock(&port->io_lock);
#endif

	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	copy_to_user(arg, payload, ARG_SIZE);
	kfree(payload);
	return 0;
}

int cuda_host_unregister(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	struct scatterlist *sgs[2], arg_sg;
	int num_out = 0, num_in = 0;
	struct virtio_uvm_page *pages= NULL;
	struct sg_table *st;
	
	func();
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}

	pages = find_pages_by_addr(payload->src, &port->page);
	if(!pages) {
		gldebug("Failed to find such user addr 0x%llx\n", payload->src);
		// cudaErrorHostMemoryNotRegistered      =     62,
		// put_user(62, &arg->cmd);
		kfree(payload);
		return -ENOMEM;
	}
	st = pages->st;
	/* meta data header*/
	sg_init_one(&arg_sg, payload, sizeof(*payload));
	sgs[num_out++] = &arg_sg;
	/*Data out buffer */
	sgs[num_out++] = st->sgl;

#ifdef VIRTIO_LOCK
	spin_lock(&port->io_lock);
#endif
	send_sgs_to_virtio(port, sgs, num_out, num_in);
#ifdef VIRTIO_LOCK
	spin_unlock(&port->io_lock);
#endif

	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	copy_to_user(arg, payload, ARG_SIZE);
	sg_free_table(st);
	kfree(st);
	list_del(&pages->list);
	kfree(pages);
	kfree(payload);
	return 0;
}

static int find_addr_in_mol(uint64_t ptr, struct port *port)
{
	MOL *mol, *mol2;
	list_for_each_entry_safe(mol, mol2, &port->device_mem_list, list) {
		if (ptr >= mol->addr && ptr < mol->addr+mol->size) {
			gldebug("Found addr in mol.\n");
			return 0;
		}
	}
	// pr_err("Failed to find device address 0x%llx\n", ptr);
	return -ENOMEM;
}

int cuda_memcpy_safe(VirtIOArg __user *arg, struct port *port)
{
	return 0;
}

static int cuda_memcpy_htod(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	uint32_t src_size;
	struct scatterlist *sgs[2], arg_sg, *sg;
	int num_out = 0, num_in = 0;
	struct sg_table *st;
	struct page ** page_list;
	int nr_pages=0;
	int ret = 0, i=0;
	void *addr;
	unsigned long size_left, block_size, offset;
	int overflow = 0;

	func();
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	gldebug("tid = %d, src=0x%llx, srcSize=0x%x, "
			"dst=0x%llx, dstSize=0x%x, kind=%llu\n",
			payload->tid, payload->src, payload->srcSize,
			payload->dst, payload->dstSize, payload->flag);
	src_size = payload->srcSize;
	page_list = uaddr_to_pages(payload->src, src_size, 0);
	if(!page_list) {
		// cudaErrorMemoryAllocation             =      2,
		put_user(2, &arg->cmd);
		kfree(payload);
		return -ENOMEM;
	}
	nr_pages = page_nr(payload->src, src_size);
	gldebug("nr page 0x%x\n", nr_pages);
	st = kmalloc(sizeof(*st), GFP_KERNEL);
	if(!st) {
		kvfree(page_list);
		kfree(payload);
		return -ENOMEM;
	}
	ret = sg_alloc_table_from_pages(st, page_list, nr_pages, 
				offset_in_page(payload->src), src_size, GFP_KERNEL);
	if(ret < 0) {
		pr_err("Failed to allocated sg table\n");
		kvfree(page_list);
		kfree(st);
		put_user(2, &arg->cmd);
		kfree(payload);
		return -ENOMEM;
	}
	kvfree(page_list);
	gldebug("sg nents %d\n", st->nents);
	if (unlikely(st->nents > VIRTIO_INDIRECT_NUM_MAX)) {
		pr_err("Pages num exceed %d\n", VIRTIO_INDIRECT_NUM_MAX);
		// cudaErrorMemoryAllocation             =      2,
		// put_user(2, &arg->cmd);
		// kfree(st);
		// kfree(payload);
		// return -ENOMEM;
		overflow = 1;
		if(!sg_zero_buffer(st->sgl, sg_nents(st->sgl), src_size, 0)) {
			pr_err("Fail to zero buffer size\n");
			kfree(payload);
			return -ENOMEM;
		}
		sg = st->sgl;
		st->nents = 0;
		size_left = src_size;
		offset = 0;
		while(size_left) {
			block_size = (size_left > CHUNK_SIZE)? CHUNK_SIZE: size_left;
			addr = kmalloc(block_size, GFP_KERNEL);
			while(!addr) {
				block_size /=2;
				if(unlikely(block_size < PAGE_SIZE)) {
					pr_err("[ERROR] Failed to allocate memory.\n");
					// cudaErrorInvalidValue                 =     11,
					put_user(11, &arg->cmd);
					kfree(payload);
					return -ENOMEM;
				}
				addr = kmalloc(block_size, GFP_KERNEL);
			}
			// gldebug("block size 0x%lx\n", block_size);
			if (copy_from_user((void*)addr, (void*)payload->src+offset, block_size)) {
				pr_err("[ERROR] Failed to copy from user.\n");
				// cudaErrorInvalidValue                 =     11,
				put_user(11, &arg->cmd);
				kfree(payload);
				return -ENOMEM;
			}
			size_left -= block_size;
			offset += block_size;
			sg_set_page(sg, virt_to_page(addr), block_size, 0);
			st->nents++;
			if(!size_left) {
				sg_mark_end(sg);
				break;
			}
			sg = sg_next(sg);
		}
		gldebug("new sg nents %d\n", st->nents);
	}
	sg_init_one(&arg_sg, payload, sizeof(*payload));
	sgs[num_out++] = &arg_sg;
	sgs[num_out++] = st->sgl;
#ifdef VIRTIO_LOCK
	spin_lock(&port->io_lock);
#endif
	send_sgs_to_virtio(port, sgs, num_out, num_in);
#ifdef VIRTIO_LOCK
	spin_unlock(&port->io_lock);
#endif

	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	copy_to_user(arg, payload, ARG_SIZE);
	if(overflow) {
		for_each_sg(st->sgl, sg, st->nents, i) {
			if(sg_page(sg)) {
				// gldebug("Free %x, addr 0x%lx\n", i, page_to_virt(sg_page(sg)));
				kfree(page_to_virt(sg_page(sg)));
			}
		}
	}
	sg_free_table(st);
	kfree(st);
	kfree(payload);
	return 0;
}

static int cuda_memcpy_dtoh(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	uint32_t src_size;
	struct scatterlist *sgs[2], arg_sg, *sg;
	int num_out = 0, num_in = 0;
	struct sg_table *st;
	struct page ** page_list;
	int nr_pages=0;
	int ret = 0, i=0;
	void *addr;
	unsigned long size_left, block_size, offset;
	int overflow = 0;

	func();
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	gldebug("tid = %d, src=0x%llx, srcSize=0x%x, "
			"dst=0x%llx, dstSize=0x%x, kind=%llu\n",
			payload->tid, payload->src, payload->srcSize,
			payload->dst, payload->dstSize, payload->flag);
	src_size = payload->srcSize;
	page_list = uaddr_to_pages(payload->dst, src_size, 1);
	if(!page_list) {
		// cudaErrorMemoryAllocation             =      2,
		put_user(2, &arg->cmd);
		kfree(payload);
		return -ENOMEM;
	}
	nr_pages = page_nr(payload->dst, src_size);
	gldebug("nr page %x\n", nr_pages);
	st = kmalloc(sizeof(*st), GFP_KERNEL);
	if(!st) {
		kvfree(page_list);
		kfree(payload);
		return -ENOMEM;
	}
	ret = sg_alloc_table_from_pages(st, page_list, nr_pages, 
				offset_in_page(payload->dst), src_size, GFP_KERNEL);
	if(ret < 0) {
		pr_err("Failed to allocated sg table\n");
		put_user(2, &arg->cmd);
		kvfree(page_list);
		kfree(st);
		kfree(payload);
		return -ENOMEM;
	}
	kvfree(page_list);
	gldebug("sg nents %d\n", st->nents);
	if (unlikely(st->nents > VIRTIO_INDIRECT_NUM_MAX)) {
		pr_err("Pages num exceed %d\n", VIRTIO_INDIRECT_NUM_MAX);
		overflow = 1;
		if(!sg_zero_buffer(st->sgl, sg_nents(st->sgl), src_size, 0)) {
			pr_err("Fail to zero buffer size\n");
			kfree(payload);
			return -ENOMEM;
		}
		sg = st->sgl;
		st->nents = 0;
		size_left = src_size;
		
		while(size_left) {
			block_size = (size_left > CHUNK_SIZE)? CHUNK_SIZE: size_left;
			addr = kmalloc(block_size, GFP_KERNEL);
			while(!addr) {
				block_size /=2;
				if(unlikely(block_size < PAGE_SIZE)) {
					pr_err("[ERROR] Failed to allocate memory.\n");
					// cudaErrorInvalidValue                 =     11,
					put_user(11, &arg->cmd);
					kfree(payload);
					return -ENOMEM;
				}
				addr = kmalloc(block_size, GFP_KERNEL);
			}
			size_left -= block_size;
			sg_set_page(sg, virt_to_page(addr), block_size, 0);
			st->nents++;
			if(!size_left) {
				sg_mark_end(sg);
				break;
			}
			sg = sg_next(sg);
		}
		gldebug("new sg nents %d\n", st->nents);
	}
	sg_init_one(&arg_sg, payload, sizeof(*payload));
	sgs[num_out++] = &arg_sg;
	sgs[num_out+num_in++] = st->sgl;
#ifdef VIRTIO_LOCK
	spin_lock(&port->io_lock);
#endif
	send_sgs_to_virtio(port, sgs, num_out, num_in);
#ifdef VIRTIO_LOCK
	spin_unlock(&port->io_lock);
#endif

	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	copy_to_user(arg, payload, ARG_SIZE);
	if(overflow) {
		offset = 0;
		for_each_sg(st->sgl, sg, st->nents, i) {
			if(sg_page(sg)) {
				addr = page_to_virt(sg_page(sg));
				gldebug("copy to user, then free %x, addr 0x%p, len 0x%x\n", 
							i, addr, sg->length);
				if (copy_to_user((void*)payload->dst+offset, (void*)addr, sg->length)) {
					pr_err("[ERROR] Failed to copy to user.\n");
					// cudaErrorInvalidValue                 =     11,
					put_user(11, &arg->cmd);
					return -ENOMEM;
				}
				offset += sg->length;
				kfree(addr);
			}
		}
	}
	sg_free_table(st);
	kfree(st);
	kfree(payload);
	return 0;
}

static int cuda_memcpy_dtod(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

static int cuda_memcpy_htod_async(VirtIOArg __user *arg, struct port *port)
{
	
	func();
	return cuda_memcpy_htod(arg, port);
}

static int cuda_memcpy_dtoh_async(VirtIOArg __user *arg, struct port *port)
{
	func();
	return cuda_memcpy_dtoh(arg, port);
}

static int cuda_memcpy_dtod_async(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

static int cuda_memset(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

static int cuda_memcpy(VirtIOArg __user *arg, struct port *port)
{
	return -EACCES;
}

int cuda_memcpy_async(VirtIOArg __user *arg, struct port *port)
{
	return -EACCES;
}

int cuda_get_last_error(VirtIOArg __user *arg, struct port *port)
{
	return -EACCES;
}

int cuda_peek_at_last_error(VirtIOArg __user *arg, struct port *port)
{
	return -EACCES;
}

static int cuda_get_device_properties(VirtIOArg __user *arg, struct port *port)
{
	int device;
	int buf_size;
	unsigned long addr;
	struct vgpu_device *vgpu;
	func();

	if(get_user(device, &arg->flag)){
		pr_err("[ERROR] can not get device id\n");
		return -ENXIO;
	}
	if(get_user(buf_size, &arg->dstSize)){
		pr_err("[ERROR] can not get buf size\n");
		return -ENXIO;
	}
	if(get_user(addr, &arg->dst)){
		pr_err("[ERROR] can not get buf\n");
		return -ENXIO;
	}
	gldebug("device id is %d.\n", device);

	vgpu = find_gpu_by_device(port->portdev, device);
	if (!vgpu) {
		pr_err("Failed to find properties of device id %d.\n", device);
		// cudaErrorInvalidDevice                =     10,
		put_user(10, &arg->cmd);
		return -ENXIO;
	}
	gldebug("prop_size is %u.\n", vgpu->prop_size);
	copy_to_user((void __user *)addr, vgpu->prop_buf, buf_size);	
	put_user(0, &arg->cmd);
	return 0;
}

static int cuda_get_device_count(VirtIOArg __user *arg, struct port *port)
{
	func();
	put_user(0, &arg->cmd);
	gldebug("gpu count=%d\n", port->portdev->nr_vgpus);
	put_user(port->portdev->nr_vgpus, &arg->flag);
	return 0;
}

static int cuda_get_device(VirtIOArg __user *arg, struct port *port)
{
	func();
	put_user(0, &arg->cmd);
	gldebug("gpu device id=%d\n", port->device);
	put_user(port->device, &arg->flag);
	return 0;
}

static int cuda_get_device_flags(VirtIOArg __user *arg, struct port *port)
{
	struct vgpu_device *vgpu;
	func();

	vgpu = find_gpu_by_device(port->portdev, port->device);
	if (!vgpu) {
		pr_err("Failed to find properties of device id %d.\n", port->device);
		// cudaErrorInvalidDevice                =     10,
		put_user(10, &arg->cmd);
		return -EACCES;
	}
	put_user(0, &arg->cmd);
	put_user(vgpu->flags, &arg->flag);
	return 0;
}

static int cuda_set_device(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;

	func();
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	if(payload->flag < 0 || payload->flag >= port->portdev->nr_vgpus) {
		pr_err("[ERROR] device nr is not in range.\n");
		// cudaErrorInvalidDevice                =     10,
		put_user(10, &arg->cmd);
		return -EACCES;
	}
	port->device = payload->flag;
	put_user(0, &arg->cmd);
	kfree(payload);
	return 0;
}

static int cuda_set_device_flags(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	struct vgpu_device *vgpu;
	func();

	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	vgpu = find_gpu_by_device(port->portdev, port->device);
	if (!vgpu) {
		pr_err("Failed to find properties of device id %d.\n", port->device);
		// cudaErrorInvalidDevice                =     10,
		put_user(10, &arg->cmd);
		return -EACCES;
	}
	if (vgpu->initialized) {
		// cudaErrorSetOnActiveProcess           =     36,
		pr_err("cudaErrorSetOnActiveProcess of device id %d.\n", port->device);
		put_user(36, &arg->cmd);
		return -EACCES;
	}
	vgpu->flags = (unsigned int )payload->flag;
	put_user(0, &arg->cmd);
	kfree(payload);
	return 0;
}

int cuda_device_set_cache_config(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

static int cuda_device_reset(VirtIOArg __user *arg, struct port *port)
{
	struct vgpu_device *vgpu;

	func();
	vgpu = find_gpu_by_device(port->portdev, port->device);
	if (!vgpu) {
		pr_err("Failed to find properties of device id %d.\n", port->device);
		// cudaErrorInvalidDevice                =     10,
		put_user(10, &arg->cmd);
		return -EACCES;
	}
	if(!vgpu->initialized) {
		put_user(0, &arg->cmd);
		return -EACCES;
	}
	vgpu->initialized = 0;
	vgpu->flags = 0;
	return send_single_payload(arg, port);
}

static int cuda_device_synchronize(VirtIOArg __user *arg, struct port *port)
{
	struct vgpu_device *vgpu;

	func();
	vgpu = find_gpu_by_device(port->portdev, port->device);
	if (!vgpu) {
		pr_err("Failed to find properties of device id %d.\n", port->device);
		// cudaErrorInvalidDevice                =     10,
		put_user(10, &arg->cmd);
		return -EACCES;
	}
	if(!vgpu->initialized) {
		put_user(0, &arg->cmd);
		return -EACCES;
	}
	return send_single_payload(arg, port);
}

static int cuda_stream_create(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

int cuda_stream_create_with_flags(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

int cuda_stream_destroy(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

int cuda_stream_synchronize(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

int cuda_stream_wait_event(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

static int cuda_event_create(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

static int cuda_event_create_with_flags(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

static int cuda_event_destroy(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

static int cuda_event_record(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

static int cuda_event_synchronize(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

static int cuda_event_query(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

static int cuda_event_elapsed_time(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	struct scatterlist *sgs[2], arg_sg, dst_sg;
	int num_out = 0, num_in = 0;
	void *ptr;

	func();
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	ptr = kmalloc((size_t)payload->dstSize, GFP_KERNEL);
	if(!ptr) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", payload->dstSize);
		return -ENOMEM;
	}
	sg_init_one(&arg_sg, payload, sizeof(*payload));
	sgs[num_out++] = &arg_sg;
	sg_init_one(&dst_sg, ptr, payload->dstSize);
	sgs[num_out+num_in++] = &dst_sg;
#ifdef VIRTIO_LOCK
	spin_lock(&port->io_lock);
#endif
	send_sgs_to_virtio(port, sgs, num_out, num_in);
#ifdef VIRTIO_LOCK
	spin_unlock(&port->io_lock);
#endif
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	copy_to_user((void __user *)payload->dst, ptr, payload->dstSize);
	kfree(ptr);
	kfree(payload);
	return 0;
}

static int cuda_thread_synchronize(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

static int cuda_mem_get_info(VirtIOArg __user *arg, struct port *port)
{
	func();
	return send_single_payload(arg, port);
}

int cublas_create(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	void *ptr;
	func();

	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	ptr = kmalloc((size_t)payload->srcSize, GFP_KERNEL);
	if(!ptr) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", payload->srcSize);
		return -ENOMEM;
	}
	payload->dst = (uint64_t)virt_to_phys(ptr);

	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	copy_to_user((void __user *)payload->src, ptr, payload->srcSize);
	kfree(ptr);
	kfree(payload);
	return ret;
}

int cublas_destroy(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	void *ptr;

	func();
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	ptr = (VirtIOArg *)memdup_user((void*)payload->src, payload->srcSize);
	if(!ptr) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", payload->srcSize);
		return -ENOMEM;
	}
	payload->dst = (uint64_t)virt_to_phys(ptr);
	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	kfree(ptr);
	kfree(payload);
	return ret;
}

int cublas_set_vector(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	uint32_t size;
	void *ptr = NULL;
	void *h_mem = NULL;

	func();
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	size = payload->srcSize;
	payload->flag = 1;
	if(!find_page_by_addr(payload->src, port)) {
		payload->flag = 0;
		h_mem = memdup_user((const void __user *)payload->src, (size_t)size);
		if(!h_mem) {
			pr_err("[ERROR] can not malloc 0x%x memory\n", size);
			return -ENOMEM;
		}
		payload->src = (uint64_t)virt_to_phys(h_mem);
	}
	ptr = memdup_user((const void __user*)payload->param, payload->paramSize);
	if(!ptr) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", payload->paramSize);
		return -ENOMEM;
	}
	payload->param = (uint64_t)virt_to_phys(ptr);
	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	kfree(ptr);
	if(!payload->flag)
		kfree(h_mem);
	kfree(payload);
	return ret;
}

int cublas_get_vector(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	uint32_t size;
	void *ptr = NULL;
	void *h_mem = NULL;

	func();
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	size = payload->srcSize;
	payload->flag = 1;
	if(!find_page_by_addr(payload->dst, port)) {
		payload->flag = 0;
		h_mem = kmalloc(size, GFP_KERNEL);
		if(!h_mem) {
			pr_err("[ERROR] can not malloc 0x%x memory\n", size);
			return -ENOMEM;
		}
		payload->param2 = (uint64_t)virt_to_phys(h_mem);
	}
	ptr = memdup_user((const void __user*)payload->param, payload->paramSize);
	if(!ptr) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", payload->paramSize);
		return -ENOMEM;
	}
	payload->param = (uint64_t)virt_to_phys(ptr);
	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	kfree(ptr);
	if(!payload->flag) {
		copy_to_user((void __user *)payload->dst, h_mem, size);
		kfree(h_mem);
	}
	kfree(payload);
	return ret;
}

int cublas_set_stream(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int len = sizeof(VirtIOArg);
	int ret;

	func();
	payload = (VirtIOArg *)memdup_user(arg, len);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", len);
		return -ENOMEM;
	}
	ret = send_to_virtio(port, (void*)payload, len);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	kfree(payload);
	return ret;
}

int cublas_get_stream(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	func();

	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	put_user(payload->flag, &arg->flag);
	kfree(payload);
	return ret;
}

static int cublas_asum(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	void *ptr = NULL;
	uint32_t size = 0;

	func();
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	size = payload->paramSize;
	ptr = kmalloc(size, GFP_KERNEL);
	if(!ptr) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", size);
		return -ENOMEM;
	}
	payload->param2 = (uint64_t)virt_to_phys(ptr);
	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	copy_to_user((void __user *)payload->param, ptr, size);
	kfree(ptr);
	kfree(payload);
	return ret;
}

int cublas_sasum(VirtIOArg __user *arg, struct port *port)
{
	func();
	return cublas_asum(arg, port);
}

int cublas_dasum(VirtIOArg __user *arg, struct port *port)
{
	func();
	return cublas_asum(arg, port);
}

static int cublas_copy(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	func();

	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	kfree(payload);
	return ret;
}

int cublas_scopy(VirtIOArg __user *arg, struct port *port)
{
	func();
	return cublas_copy(arg, port);
}

int cublas_dcopy(VirtIOArg __user *arg, struct port *port)
{
	func();
	return cublas_copy(arg, port);
}

static int cublas_dot(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	void *ptr = NULL;
	uint32_t size = 0;
	func();

	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	size = payload->paramSize;
	ptr = kmalloc(size, GFP_KERNEL);
	if(!ptr) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", size);
		return -ENOMEM;
	}
	payload->param2 = (uint64_t)virt_to_phys(ptr);
	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	copy_to_user((void __user *)payload->param, ptr, size);
	kfree(ptr);
	kfree(payload);
	return ret;
}

int cublas_sdot(VirtIOArg __user *arg, struct port *port)
{
	func();
	return cublas_dot(arg, port);
}

int cublas_ddot(VirtIOArg __user *arg, struct port *port)
{
	func();
	return cublas_dot(arg, port);
}

static int cublas_axpy(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	void *ptr = NULL;
	func();

	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	ptr = memdup_user((const void __user*)payload->param, payload->paramSize);
	if(!ptr) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", payload->paramSize);
		return -ENOMEM;
	}
	payload->param = (uint64_t)virt_to_phys(ptr);
	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	kfree(ptr);
	kfree(payload);
	return ret;
}

int cublas_saxpy(VirtIOArg __user *arg, struct port *port)
{
	func();
	return cublas_axpy(arg, port);
}

int cublas_daxpy(VirtIOArg __user *arg, struct port *port)
{
	func();
	return cublas_axpy(arg, port);
}

static int cublas_scal(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	void *ptr = NULL;
	func();

	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	ptr = memdup_user((const void __user*)payload->param, payload->paramSize);
	if(!ptr) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", payload->paramSize);
		return -ENOMEM;
	}
	payload->param = (uint64_t)virt_to_phys(ptr);
	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	kfree(ptr);
	kfree(payload);
	return ret;
}

int cublas_sscal(VirtIOArg __user *arg, struct port *port)
{
	func();
	return cublas_scal(arg, port);
}

int cublas_dscal(VirtIOArg __user *arg, struct port *port)
{
	func();
	return cublas_scal(arg, port);
}

static int cublas_gemv(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	void *ptr;

	func();
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	ptr = memdup_user((const void __user*)payload->param, payload->paramSize);
	if(!ptr) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", payload->paramSize);
		return -ENOMEM;
	}
	payload->param = (uint64_t)virt_to_phys(ptr);
	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	kfree(ptr);
	kfree(payload);
	return ret;
}

int cublas_sgemv(VirtIOArg __user *arg, struct port *port)
{
	func();
	return cublas_gemv(arg, port);
}

int cublas_dgemv(VirtIOArg __user *arg, struct port *port)
{
	func();
	return cublas_gemv(arg, port);
}

static int cublas_gemm(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int len = sizeof(VirtIOArg);
	int ret;
	void *ptr;

	func();
	payload = (VirtIOArg *)memdup_user(arg, len);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", len);
		return -ENOMEM;
	}

	ptr = memdup_user((const void __user*)payload->param, payload->paramSize);
	if(!ptr) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", payload->paramSize);
		return -ENOMEM;
	}
	payload->param = (uint64_t)virt_to_phys(ptr);
	ret = send_to_virtio(port, (void*)payload, len);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	kfree(ptr);
	kfree(payload);
	return ret;
}

int cublas_sgemm(VirtIOArg __user *arg, struct port *port)
{
	func();
	return cublas_gemm(arg, port);
}

int cublas_dgemm(VirtIOArg __user *arg, struct port *port)
{
	func();
	return cublas_gemm(arg, port);
}

int cublas_set_matrix(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	uint32_t size;
	void *ptr = NULL;
	void *h_mem = NULL;

	func();
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	size = payload->srcSize;
	payload->flag = 1;
	if(!find_page_by_addr(payload->src, port)) {
		payload->flag = 0;
		h_mem = memdup_user((const void __user *)payload->src, (size_t)size);
		if(!h_mem) {
			pr_err("[ERROR] can not malloc 0x%x memory\n", size);
			return -ENOMEM;
		}
		payload->src = (uint64_t)virt_to_phys(h_mem);
	}
	ptr = memdup_user((const void __user*)payload->param, payload->paramSize);
	if(!ptr) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", payload->paramSize);
		return -ENOMEM;
	}
	payload->param = (uint64_t)virt_to_phys(ptr);
	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	kfree(ptr);
	if(!payload->flag)
		kfree(h_mem);
	kfree(payload);
	return ret;
}

int cublas_get_matrix(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	uint32_t size;
	void *ptr = NULL;
	void *h_mem = NULL;

	func();
	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	size = payload->srcSize;
	payload->flag = 1;
	if(!find_page_by_addr(payload->dst, port)) {
		payload->flag = 0;
		h_mem = kmalloc(size, GFP_KERNEL);
		if(!h_mem) {
			pr_err("[ERROR] can not malloc 0x%x memory\n", size);
			return -ENOMEM;
		}
		payload->param2 = (uint64_t)virt_to_phys(h_mem);
	}
	ptr = memdup_user((const void __user*)payload->param, payload->paramSize);
	if(!ptr) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", payload->paramSize);
		return -ENOMEM;
	}
	payload->param = (uint64_t)virt_to_phys(ptr);
	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	kfree(ptr);
	if(!payload->flag) {
		copy_to_user((void __user *)payload->dst, h_mem, size);
		kfree(h_mem);
	}
	kfree(payload);
	return ret;
}

static int curand_create_generator_v2(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	func();

	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	put_user(payload->flag, &arg->flag);
	kfree(payload);
	return ret;
}

int curand_create_generator(VirtIOArg __user *arg, struct port *port)
{
	func();
	return curand_create_generator_v2(arg, port);
}

int curand_create_generator_host(VirtIOArg __user *arg, struct port *port)
{
	func();
	return curand_create_generator_v2(arg, port);
}

static int curand_send_basic(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	func();

	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	kfree(payload);
	return ret;
}

int curand_generate(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	uint32_t size;
	void *h_mem = NULL;
	func();

	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	size = payload->dstSize;
	if(find_addr_in_mol(payload->dst, port) == 0) {
		payload->flag = 0;
	} else {
		if(!find_page_by_addr(payload->dst, port)) {
			payload->flag 	= 1;
			h_mem = kmalloc(size, GFP_KERNEL);
			if(!h_mem) {
				pr_err("[ERROR] can not malloc 0x%x memory\n", size);
				return -ENOMEM;
			}
			payload->param2 = (uint64_t)virt_to_phys(h_mem);
		} else {
			payload->flag 	= 2;
		}
	}
	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	if(payload->flag == 1) {
		copy_to_user((void __user *)payload->dst, h_mem, size);
		kfree(h_mem);
	}
	kfree(payload);
	return ret;
}

int curand_generate_normal_v2(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	void *ptr;
	uint32_t size;
	void *h_mem = NULL;
	func();

	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	size = payload->dstSize;
	if(find_addr_in_mol(payload->dst, port) == 0) {
		payload->flag = 0;
	} else {
		if(!find_page_by_addr(payload->dst, port)) {
			payload->flag 	= 1;
			h_mem = kmalloc(size, GFP_KERNEL);
			if(!h_mem) {
				pr_err("[ERROR] can not malloc 0x%x memory\n", size);
				return -ENOMEM;
			}
			payload->param2 = (uint64_t)virt_to_phys(h_mem);
		} else {
			payload->flag 	= 2;
		}
	}
	ptr = memdup_user((const void __user*)payload->param, payload->paramSize);
	if(!ptr) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", payload->paramSize);
		return -ENOMEM;
	}
	payload->param = (uint64_t)virt_to_phys(ptr);
	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	if(payload->flag == 1) {
		copy_to_user((void __user *)payload->dst, h_mem, size);
		kfree(h_mem);
	}
	kfree(payload);
	return ret;
}

int curand_generate_normal(VirtIOArg __user *arg, struct port *port)
{
	func();
	return curand_generate_normal_v2(arg, port);
}

int curand_generate_normal_double(VirtIOArg __user *arg, struct port *port)
{
	func();
	return curand_generate_normal_v2(arg, port);
}

int curand_generate_uniform_v2(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	int ret;
	uint32_t size;
	void *h_mem = NULL;
	func();

	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	size = payload->dstSize;
	if(find_addr_in_mol(payload->dst, port) == 0) {
		payload->flag = 0;
	} else {
		if(!find_page_by_addr(payload->dst, port)) {
			payload->flag 	= 1;
			h_mem = kmalloc(size, GFP_KERNEL);
			if(!h_mem) {
				pr_err("[ERROR] can not malloc 0x%x memory\n", size);
				return -ENOMEM;
			}
			payload->param2 = (uint64_t)virt_to_phys(h_mem);
		} else {
			payload->flag 	= 2;
		}
	}
	ret = send_to_virtio(port, (void*)payload, ARG_SIZE);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	put_user(payload->cmd, &arg->cmd);
	if(payload->flag == 1) {
		copy_to_user((void __user *)payload->dst, h_mem, size);
		kfree(h_mem);
	}
	kfree(payload);
	return ret;
}

int curand_generate_uniform(VirtIOArg __user *arg, struct port *port)
{
	func();
	return curand_generate_uniform_v2(arg, port);
}

int curand_generate_uniform_double(VirtIOArg __user *arg, struct port *port)
{
	func();
	return curand_generate_uniform_v2(arg, port);
}

int curand_destroy_generator(VirtIOArg __user *arg, struct port *port)
{
	func();
	return curand_send_basic(arg, port);
}

int curand_set_generator_offset(VirtIOArg __user *arg, struct port *port)
{
	func();
	return curand_send_basic(arg, port);
}

int curand_set_pseudorandom_seed(VirtIOArg __user *arg, struct port *port)
{
	func();
	return curand_send_basic(arg, port);
}

/**** SGX ************************************************************/
static int sgx_proc_msg0(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	struct scatterlist *sgs[2], arg_sg, gva_sg;
	int num_out=0, num_in=0;
	void *ptr;
	unsigned long size=0;
	func();

	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	size = payload->dstSize;
	ptr = kmalloc(size, GFP_KERNEL);
	if(!ptr) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", size);
		return -ENOMEM;
	}
	sg_init_one(&arg_sg, payload, sizeof(*payload));
	sgs[num_out++] = &arg_sg;
	sg_init_one(&gva_sg, ptr, size);
	sgs[num_out+num_in++] = &gva_sg;
	send_sgs_to_virtio(port, sgs, num_out, num_in);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	copy_to_user(arg, payload, ARG_SIZE);
	if(payload->paramSize > size) {
		pr_err("[ERROR] Return bufer overflow!\n");
		return -ENOMEM;
	}
	copy_to_user((void __user *)payload->dst, ptr, payload->paramSize);
	kfree(ptr);
	kfree(payload);
	return 0;
}

static int sgx_proc_msg1(VirtIOArg __user *arg, struct port *port)
{
	VirtIOArg *payload;
	struct scatterlist *sgs[3], arg_sg, req_sg, resp_sg;
	int num_out=0, num_in=0;
	void *req, *resp;
	func();

	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);
		return -ENOMEM;
	}
	req = memdup_user((const void __user*)payload->src, payload->srcSize);
	if(!req) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", payload->srcSize);
		return -ENOMEM;
	}
	resp = kmalloc((size_t)payload->dstSize, GFP_KERNEL);
	if(!resp) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", payload->dstSize);
		return -ENOMEM;
	}
	sg_init_one(&arg_sg, payload, sizeof(*payload));
	sgs[num_out++] = &arg_sg;
	sg_init_one(&req_sg, req, payload->srcSize);
	sgs[num_out++] = &req_sg;
	sg_init_one(&resp_sg, resp, payload->dstSize);
	sgs[num_out+num_in++] = &resp_sg;
	send_sgs_to_virtio(port, sgs, num_out, num_in);
	gldebug("[+] now analyse return buf\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	copy_to_user(arg, payload, ARG_SIZE);
	if(payload->paramSize > payload->dstSize) {
		pr_err("[ERROR] Return bufer overflow!\n");
		return -ENOMEM;
	}
	copy_to_user((void __user *)payload->dst, resp, payload->paramSize);
	kfree(req);
	kfree(resp);
	kfree(payload);
	return 0;
}

static int sgx_proc_msg3(VirtIOArg __user *arg, struct port *port)
{
	func();
	return sgx_proc_msg1(arg, port);
}

int cuda_gpa_to_hva(VirtIOArg __user *arg, struct port *port)
{
	void *gva, *gpa;
	VirtIOArg *payload;
	uint32_t from_size;
	int ret=0;
	func();

	payload = (VirtIOArg *)memdup_user(arg, ARG_SIZE);
	if(!payload) {
		pr_err("[ERROR] can not malloc 0x%lx memory\n", ARG_SIZE);	
		return -ENOMEM;
	}
	from_size = payload->srcSize;

	gva = memdup_user((const void __user *)payload->src, (size_t)from_size);
	if(!gva) {
		pr_err("[ERROR] can not malloc 0x%x memory\n", from_size);	
		return -ENOMEM;
	}
	gpa = (void*)virt_to_phys(gva);
	payload->src = (uint64_t)gpa;
	gldebug("*gva=%d, &gva=0x%p, gpa=0x%p \n",  *(int*)gva, gva, gpa);
	
	ret = send_to_virtio(port, (void *)payload, ARG_SIZE);
	gldebug("[+]== now analyse return buf ==\n");
	gldebug("[+] arg->cmd = %d\n", payload->cmd);
	gldebug("[+] val=%d, virt= 0x%p, phys= 0x%p.\n", \
		*(int*)phys_to_virt((phys_addr_t)gpa), gva, gpa);
	copy_to_user((void __user *)payload->src, gva, from_size);
	put_user(payload->cmd, &arg->cmd);
	kfree(gva);
	kfree(payload);
	return ret;
}

static long port_fops_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	bool nonblock;
	struct port *port;
	int ret=0;
	void __user *argp = (void __user *)arg;
	func();

	port = filp->private_data;
	gldebug("port->id=%d\n", port->id);
	if (!argp)
		return -EINVAL;

	nonblock = filp->f_flags & O_NONBLOCK;
	ret = wait_port_writable(port, nonblock);
	if (ret < 0)
		return ret;

	gldebug("in cpu: %d\n",smp_processor_id());
	gldebug("cmd ioctl nr = %u!\n", _IOC_NR(cmd));
	switch(cmd) {
		case VIRTIO_IOC_HELLO:
			ret = cuda_gpa_to_hva((VirtIOArg __user*)argp, port);
			break;
		case VIRTIO_IOC_PRIMARYCONTEXT:
			ret = cuda_primarycontext((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_REGISTERFATBINARY:
			ret = cuda_register_fatbinary((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_UNREGISTERFATBINARY:
			ret = cuda_unregister_fatbinary((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_REGISTERFUNCTION:
			ret = cuda_register_function((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_LAUNCH:
			ret = cuda_launch((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_LAUNCH_KERNEL:
			ret = cuda_launch_kernel((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_MALLOC:
			ret = cuda_malloc((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_MEMCPY:
			ret = cuda_memcpy((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_MEMCPY_HTOD:
			ret = cuda_memcpy_htod((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_MEMCPY_DTOH:
			ret = cuda_memcpy_dtoh((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_MEMCPY_DTOD:
			ret = cuda_memcpy_dtod((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_MEMCPY_HTOD_ASYNC:
			ret = cuda_memcpy_htod_async((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_MEMCPY_DTOH_ASYNC:
			ret = cuda_memcpy_dtoh_async((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_MEMCPY_DTOD_ASYNC:
			ret = cuda_memcpy_dtod_async((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_SGX_MEMCPY:
			ret = cuda_memcpy_safe((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_FREE:
			ret = cuda_free((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_GETDEVICE:
			ret = cuda_get_device((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_GETDEVICEPROPERTIES:
			ret = cuda_get_device_properties((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_SETDEVICE:
			ret = cuda_set_device((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_DEVICESETCACHECONFIG:
			ret = cuda_device_set_cache_config((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_GETDEVICECOUNT:
			ret = cuda_get_device_count((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_DEVICERESET:
			ret = cuda_device_reset((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_STREAMCREATE:
			ret = cuda_stream_create((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_STREAMCREATEWITHFLAGS:
			ret = cuda_stream_create_with_flags((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_STREAMDESTROY:
			ret = cuda_stream_destroy((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_EVENTCREATE:
			ret = cuda_event_create((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_EVENTDESTROY:
			ret = cuda_event_destroy((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_EVENTQUERY:
			ret = cuda_event_query((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_THREADSYNCHRONIZE:
			ret = cuda_thread_synchronize((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_EVENTSYNCHRONIZE:
			ret = cuda_event_synchronize((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_EVENTELAPSEDTIME:
			ret = cuda_event_elapsed_time((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_EVENTRECORD:
			ret = cuda_event_record((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_GETLASTERROR:
			ret = cuda_get_last_error((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_PEEKATLASTERROR:
			ret = cuda_peek_at_last_error((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_MEMCPY_ASYNC:
			ret = cuda_memcpy_async((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_MEMSET:
			ret = cuda_memset((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_DEVICESYNCHRONIZE:
			ret = cuda_device_synchronize((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_EVENTCREATEWITHFLAGS:
			ret = cuda_event_create_with_flags((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_MEMGETINFO:
			ret = cuda_mem_get_info((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_SETDEVICEFLAGS:
			ret = cuda_set_device_flags((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_GETDEVICEFLAGS:
			ret = cuda_get_device_flags((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_HOSTREGISTER:
			ret = cuda_host_register((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_HOSTUNREGISTER:
			ret = cuda_host_unregister((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_MALLOCHOST:
			ret = cuda_malloc_host((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_FREEHOST:
			ret = cuda_free_host((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_MEMCPYTOSYMBOL:
			ret = cuda_memcpy_to_symbol((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_MEMCPYFROMSYMBOL:
			ret = cuda_memcpy_from_symbol((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_REGISTERVAR:
			ret = cuda_register_var((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_STREAMWAITEVENT:
			ret = cuda_stream_wait_event((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_STREAMSYNCHRONIZE:
			ret = cuda_stream_synchronize((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_CREATE:
			ret = cublas_create((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_DESTROY:
			ret = cublas_destroy((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_SETVECTOR:
			ret = cublas_set_vector((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_GETVECTOR:
			ret = cublas_get_vector((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_SETMATRIX:
			ret = cublas_set_matrix((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_GETMATRIX:
			ret = cublas_get_matrix((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_SETSTREAM:
			ret = cublas_set_stream((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_GETSTREAM:
			ret = cublas_get_stream((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_SASUM:
			ret = cublas_sasum((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_DASUM:
			ret = cublas_dasum((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_SAXPY:
			ret = cublas_saxpy((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_DAXPY:
			ret = cublas_daxpy((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_SCOPY:
			ret = cublas_scopy((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_DCOPY:
			ret = cublas_dcopy((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_SDOT:
			ret = cublas_sdot((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_DDOT:
			ret = cublas_ddot((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_SSCAL:
			ret = cublas_sscal((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_DSCAL:
			ret = cublas_dscal((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_SGEMV:
			ret = cublas_sgemv((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_DGEMV:
			ret = cublas_dgemv((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_SGEMM:
			ret = cublas_sgemm((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CUBLAS_DGEMM:
			ret = cublas_dgemm((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CURAND_CREATEGENERATOR:
			ret = curand_create_generator((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CURAND_CREATEGENERATORHOST:
			ret = curand_create_generator_host((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CURAND_GENERATE:
			ret = curand_generate((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CURAND_GENERATENORMAL:
			ret = curand_generate_normal((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CURAND_GENERATENORMALDOUBLE:
			ret = curand_generate_normal_double((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CURAND_GENERATEUNIFORM:
			ret = curand_generate_uniform((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CURAND_GENERATEUNIFORMDOUBLE:
			ret = curand_generate_uniform_double((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CURAND_DESTROYGENERATOR:
			ret = curand_destroy_generator((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CURAND_SETGENERATOROFFSET:
			ret = curand_set_generator_offset((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_CURAND_SETPSEUDORANDOMSEED:
			ret = curand_set_pseudorandom_seed((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_SGX_MSG0:
			ret = sgx_proc_msg0((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_SGX_MSG1:
			ret = sgx_proc_msg1((VirtIOArg __user*)arg, port);
			break;
		case VIRTIO_IOC_SGX_MSG3:
			ret = sgx_proc_msg3((VirtIOArg __user*)arg, port);
			break;
		default:
			pr_err("[#] illegel VIRTIO ioctl nr = %u!\n", \
				_IOC_NR(cmd));
			return -EINVAL;
	}
	return ret;
}

static void virtio_cuda_device_mmap(struct vm_area_struct *vma)
{
	func();
	gldebug("VMA open, virt %lx, end %lx, phys %lx\n",
		vma->vm_start, vma->vm_end, vma->vm_pgoff << PAGE_SHIFT);
}

static void virtio_cuda_device_munmap(struct vm_area_struct *vma)
{
	struct sg_table *st = vma->vm_private_data;
	int i=0;
	struct scatterlist *sg;

	func();
	gldebug("munmap 0x%lx-0x%lx\n", vma->vm_start, vma->vm_end);
	for_each_sg(st->sgl, sg, st->nents, i) {
		if(sg_page(sg)) {
			__free_pages(sg_page(sg), get_order(sg->length));
		}
	}
	sg_free_table(st);
	kfree(st);
}

static struct vm_operations_struct cuda_mmap_ops = {
	.open = virtio_cuda_device_mmap,
	.close = virtio_cuda_device_munmap,
};

static int port_fops_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct port *port = NULL;
	int order = 0;
	size_t size = vma->vm_end - vma->vm_start;
	unsigned long offset, size_left, block_size;
	int i = 0;
	struct sg_table *st;
	struct scatterlist *sg;
	struct page* page;
	unsigned long addr;
	int ret = 0;


	func();
	port = filp->private_data;
	gldebug("port->id=%d\n", port->id);
	// max order is 11
	size_left = PAGE_ALIGN(size);
	st = kmalloc(sizeof(*st), GFP_KERNEL);
	if(!st)
		return -ENOMEM;
	if(sg_alloc_table(st, size_left>>PAGE_SHIFT, GFP_KERNEL)) {
		kfree(st);
		return -ENOMEM;
	}
	sg = st->sgl;
	st->nents = 0;
	gldebug("size %lx round up aligned by PAGE_SIZE is %lx\n", size_left, size);
	offset=0;
	while(size_left) {
		block_size = (size_left > CHUNK_SIZE)? CHUNK_SIZE: size_left;
		order = get_order(block_size);
		// gldebug("block size 0x%lx order is %d\n", block_size, order);
		page = alloc_pages(GFP_KERNEL, order);
		while(!page) {
			block_size /=2;
			if(block_size < PAGE_SIZE) {
				pr_err("[ERROR] Failed to allocate memory.\n");
				goto error;
			}
			order = get_order(block_size);
			page = alloc_pages(GFP_KERNEL, order);
		}

		// if(!PageCompand(page))
		split_page(page, order);
		addr = vma->vm_start + offset;
		for (i=0; i< 1<<order; i++) {
			ret = vm_insert_page(vma, addr, page+i);
			if( ret ) {
				pr_err("Failed to mmap kaddr[%x] to userspace.ret %d\n", i, ret);
				goto error;
			}
			addr += PAGE_SIZE;
			sg_set_page(sg, page+i, PAGE_SIZE, 0);
			st->nents++;
			if (unlikely(addr >= vma->vm_end))
				break;
			sg = sg_next(sg);
		}
		offset += block_size;
		size_left -= block_size;
		if(!size_left) {
			sg_mark_end(sg);
			break;
		}
	}
	vma->vm_ops = &cuda_mmap_ops;
	vma->vm_private_data = st;
	virtio_cuda_device_mmap(vma);
	return 0;
error:
	pr_err("Failed to mmap.\n");
	sg_set_page(sg, NULL, 0, 0);
	sg_mark_end(sg);
	for_each_sg(st->sgl, sg, st->nents, i) {
		if(sg_page(sg))
			__free_pages(sg_page(sg), get_order(sg->length));
	}
	sg_free_table(st);
	kfree(st);
	return -ENOMEM;
}

/*
 * The file operations that we support: programs in the guest can open
 * a console device, read from it, write to it, poll for data and
 * close it, control it using ioctl.  The devices are at
 *   /dev/cudaport<device number>p<port number>
 */
static const struct file_operations port_fops = {
	.owner = THIS_MODULE,
	.open  = port_fops_open,
	.read  = port_fops_read,
	.write = port_fops_write,
	.unlocked_ioctl = port_fops_ioctl,
	.splice_write = port_fops_splice_write,
	.poll  = port_fops_poll,
	.release = port_fops_release,
	.fasync = port_fops_fasync,
	.llseek = no_llseek,
	.mmap = port_fops_mmap,
};

static int init_port_console(struct port *port)
{
	int ret;

	/*
	 * The Host's telling us this port is a console port.  Hook it
	 * up with an hvc console.
	 *
	 * To set up and manage our virtual console, we call
	 * hvc_alloc().
	 *
	 * The first argument of hvc_alloc() is the virtual console
	 * number.  The second argument is the parameter for the
	 * notification mechanism (like irq number).  We currently
	 * leave this as zero, virtqueues have implicit notifications.
	 *
	 * The third argument is a "struct hv_ops" containing the
	 * put_chars() get_chars(), notifier_add() and notifier_del()
	 * pointers.  The final argument is the output buffer size: we
	 * can do any size, so we put PAGE_SIZE here.
	 */
	port->cons.vtermno = pdrvdata.next_vtermno;

	// port->cons.hvc = hvc_alloc(port->cons.vtermno, 0, &hv_ops, PAGE_SIZE);
	if (IS_ERR(port->cons.hvc)) {
		ret = PTR_ERR(port->cons.hvc);
		dev_err(port->dev,
			"error %d allocating hvc for port\n", ret);
		port->cons.hvc = NULL;
		return ret;
	}
	spin_lock_irq(&pdrvdata_lock);
	pdrvdata.next_vtermno++;
	list_add_tail(&port->cons.list, &pdrvdata.consoles);
	spin_unlock_irq(&pdrvdata_lock);
	port->guest_connected = true;

	/*
	 * Start using the new console output if this is the first
	 * console to come up.
	 */
	if (early_put_chars)
		early_put_chars = NULL;

	/* Notify host of port being opened */
	send_control_msg(port, VIRTIO_CONSOLE_PORT_OPEN, 1);

	return 0;
}

static ssize_t show_port_name(struct device *dev,
			      struct device_attribute *attr, char *buffer)
{
	struct port *port;

	port = dev_get_drvdata(dev);

	return sprintf(buffer, "%s\n", port->name);
}

static DEVICE_ATTR(name, S_IRUGO, show_port_name, NULL);

static struct attribute *port_sysfs_entries[] = {
	&dev_attr_name.attr,
	NULL
};

static const struct attribute_group port_attribute_group = {
	.name = NULL,		/* put in device directory */
	.attrs = port_sysfs_entries,
};

static ssize_t debugfs_read(struct file *filp, char __user *ubuf,
			    size_t count, loff_t *offp)
{
	struct port *port;
	char *buf;
	ssize_t ret, out_offset, out_count;

	out_count = 1024;
	buf = kmalloc(out_count, GFP_KERNEL);
	if (!buf)
		return -ENOMEM;

	port = filp->private_data;
	out_offset = 0;
	out_offset += snprintf(buf + out_offset, out_count,
			       "name: %s\n", port->name ? port->name : "");
	out_offset += snprintf(buf + out_offset, out_count - out_offset,
			       "guest_connected: %d\n", port->guest_connected);
	out_offset += snprintf(buf + out_offset, out_count - out_offset,
			       "host_connected: %d\n", port->host_connected);
	out_offset += snprintf(buf + out_offset, out_count - out_offset,
			       "outvq_full: %d\n", port->outvq_full);
	out_offset += snprintf(buf + out_offset, out_count - out_offset,
			       "bytes_sent: %lu\n", port->stats.bytes_sent);
	out_offset += snprintf(buf + out_offset, out_count - out_offset,
			       "bytes_received: %lu\n",
			       port->stats.bytes_received);
	out_offset += snprintf(buf + out_offset, out_count - out_offset,
			       "bytes_discarded: %lu\n",
			       port->stats.bytes_discarded);
	out_offset += snprintf(buf + out_offset, out_count - out_offset,
			       "is_console: %s\n",
			       is_console_port(port) ? "yes" : "no");
	out_offset += snprintf(buf + out_offset, out_count - out_offset,
			       "console_vtermno: %u\n", port->cons.vtermno);

	ret = simple_read_from_buffer(ubuf, count, offp, buf, out_offset);
	kfree(buf);
	return ret;
}

static const struct file_operations port_debugfs_ops = {
	.owner = THIS_MODULE,
	.open  = simple_open,
	.read  = debugfs_read,
};


static unsigned int fill_queue(struct virtqueue *vq, spinlock_t *lock)
{
	struct port_buffer *buf;
	unsigned int nr_added_bufs;
	int ret;

	nr_added_bufs = 0;
	do {
		buf = alloc_buf(vq, PAGE_SIZE, 0);
		if (!buf)
			break;

		spin_lock_irq(lock);
		ret = add_inbuf(vq, buf);
		if (ret < 0) {
			spin_unlock_irq(lock);
			free_buf(buf, true);
			break;
		}
		nr_added_bufs++;
		spin_unlock_irq(lock);
	} while (ret > 0);

	return nr_added_bufs;
}

static void send_sigio_to_port(struct port *port)
{
	if (port->async_queue && port->guest_connected)
		kill_fasync(&port->async_queue, SIGIO, POLL_OUT);
}




static int cuda_proc_val_show(struct seq_file *file, void *v)
{
	uint32_t *val = file->private;
	seq_printf(file, "%u", *val);
	return 0;
}

static int test_proc_open(struct inode *inode, struct file *file)
{
	return single_open(file, cuda_proc_val_show, PDE_DATA(inode));
}

static const struct file_operations proc_virt_dev_count_fops = {
	.owner = THIS_MODULE,
	.open  = test_proc_open,
	.read  = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

static int proc_file_create(struct ports_device *portdev)
{
	char name[256];
	if (pdrvdata.proc_dir) {
		/*virt_device_count*/
		sprintf(name, "virtual_device_count");
		portdev->proc_virt_dev_count = proc_create_data(name, S_IRUGO, 
					pdrvdata.proc_dir, &proc_virt_dev_count_fops,
					&portdev->nr_ports);
		if(!portdev->proc_virt_dev_count){
			pr_err("Failed to create /proc/virtio-cuda/%s\n", name);
			return -EINVAL;
		}
	}
	return 0;
}

static int add_port(struct ports_device *portdev, u32 id)
{
	char debugfs_name[16];
	struct port *port;
	struct port_buffer *buf;
	dev_t devt;
	unsigned int nr_added_bufs;
	int err;
	func();

	port = kmalloc(sizeof(*port), GFP_KERNEL);
	if (!port) {
		err = -ENOMEM;
		goto fail;
	}
	kref_init(&port->kref);

	port->portdev = portdev;
	port->id = id;

	port->name = NULL;
	port->inbuf = NULL;
	port->cons.hvc = NULL;
	port->async_queue = NULL;


	port->host_connected = port->guest_connected = false;
	port->stats = (struct port_stats) { 0 };

	port->outvq_full = false;

	port->in_vq = portdev->in_vqs[port->id];
	port->out_vq = portdev->out_vqs[port->id];

	port->cdev = cdev_alloc();
	if (!port->cdev) {
		dev_err(&port->portdev->vdev->dev, "Error allocating cdev\n");
		err = -ENOMEM;
		goto free_port;
	}
	port->cdev->ops = &port_fops;

	devt = MKDEV(portdev->chr_major, id);
	err = cdev_add(port->cdev, devt, 1);
	if (err < 0) {
		dev_err(&port->portdev->vdev->dev,
			"Error %d adding cdev for port %u\n", err, id);
		goto free_cdev;
	}
	port->dev = device_create(pdrvdata.class, &port->portdev->vdev->dev,
				  devt, port, "cudaport%up%u",
				  port->portdev->vdev->index, id);
	if (IS_ERR(port->dev)) {
		err = PTR_ERR(port->dev);
		dev_err(&port->portdev->vdev->dev,
			"Error %d creating device for port %u\n",
			err, id);
		goto free_cdev;
	}

	spin_lock_init(&port->inbuf_lock);
	spin_lock_init(&port->outvq_lock);
	init_waitqueue_head(&port->waitqueue);

	/* Fill the in_vq with buffers so the host can send us data. */
	nr_added_bufs = fill_queue(port->in_vq, &port->inbuf_lock);
	if (!nr_added_bufs) {
		dev_err(port->dev, "Error allocating inbufs\n");
		err = -ENOMEM;
		goto free_device;
	}

	if (is_rproc_serial(port->portdev->vdev))
		/*
		 * For rproc_serial assume remote processor is connected.
		 * rproc_serial does not want the console port, only
		 * the generic port implementation.
		 */
		port->host_connected = true;
	else if (!use_multiport(port->portdev)) {
		/*
		 * If we're not using multiport support,
		 * this has to be a console port.
		 */
		err = init_port_console(port);
		if (err)
			goto free_inbufs;
	}

	spin_lock_irq(&portdev->ports_lock);
	list_add_tail(&port->list, &port->portdev->ports);
	portdev->nr_ports++;
	spin_unlock_irq(&portdev->ports_lock);

	port->device = 0;
	INIT_LIST_HEAD(&port->device_mem_list);
	INIT_LIST_HEAD(&port->guest_mem_list);
	INIT_LIST_HEAD(&port->page);
	spin_lock_init(&port->io_lock);
	/*
	 * Tell the Host we're set so that it can send us various
	 * configuration parameters for this port (eg, port name,
	 * caching, whether this is a console port, etc.)
	 */
	send_control_msg(port, VIRTIO_CONSOLE_PORT_READY, 1);
	
	if (pdrvdata.debugfs_dir) {
		/*
		 * Finally, create the debugfs file that we can use to
		 * inspect a port's state at any time
		 */
		snprintf(debugfs_name, sizeof(debugfs_name), "cudaport%up%u",
			 port->portdev->vdev->index, id);
		port->debugfs_file = debugfs_create_file(debugfs_name, 0444,
							 pdrvdata.debugfs_dir,
							 port,
							 &port_debugfs_ops);
	}
	return 0;

free_inbufs:
	while ((buf = virtqueue_detach_unused_buf(port->in_vq)))
		free_buf(buf, true);
free_device:
	device_destroy(pdrvdata.class, port->dev->devt);
free_cdev:
	cdev_del(port->cdev);
free_port:
	kfree(port);
fail:
	/* The host might want to notify management sw about port add failure */
	__send_control_msg(portdev, id, VIRTIO_CONSOLE_PORT_READY, 0);
	return err;
}

/* No users remain, remove all port-specific data. */
static void remove_port(struct kref *kref)
{
	struct port *port;

	port = container_of(kref, struct port, kref);

	kfree(port);
}

static void remove_port_data(struct port *port)
{
	struct port_buffer *buf;
	
	spin_lock_irq(&port->inbuf_lock);
	/* Remove unused data this port might have received. */
	discard_port_data(port);
	spin_unlock_irq(&port->inbuf_lock);

	/* Remove buffers we queued up for the Host to send us data in. */
	do {
		spin_lock_irq(&port->inbuf_lock);
		buf = virtqueue_detach_unused_buf(port->in_vq);
		spin_unlock_irq(&port->inbuf_lock);
		if (buf)
			free_buf(buf, true);
	} while (buf);

	spin_lock_irq(&port->outvq_lock);
	reclaim_consumed_buffers(port);
	spin_unlock_irq(&port->outvq_lock);

	/* Free pending buffers from the out-queue. */
	do {
		spin_lock_irq(&port->outvq_lock);
		buf = virtqueue_detach_unused_buf(port->out_vq);
		spin_unlock_irq(&port->outvq_lock);
		if (buf)
			free_buf(buf, true);
	} while (buf);
}

/*
 * Port got unplugged.  Remove port from portdev's list and drop the
 * kref reference.  If no userspace has this port opened, it will
 * result in immediate removal the port.
 */
static void unplug_port(struct port *port)
{
	spin_lock_irq(&port->portdev->ports_lock);
	list_del(&port->list);
	list_del(&port->device_mem_list);
	list_del(&port->guest_mem_list);
	port->portdev->nr_ports--;
	spin_unlock_irq(&port->portdev->ports_lock);

	spin_lock_irq(&port->inbuf_lock);
	if (port->guest_connected) {
		/* Let the app know the port is going down. */
		send_sigio_to_port(port);

		/* Do this after sigio is actually sent */
		port->guest_connected = false;
		port->host_connected = false;

		wake_up_interruptible(&port->waitqueue);
	}
	spin_unlock_irq(&port->inbuf_lock);

	if (is_console_port(port)) {
		spin_lock_irq(&pdrvdata_lock);
		list_del(&port->cons.list);
		spin_unlock_irq(&pdrvdata_lock);
		// hvc_remove(port->cons.hvc);
	}

	remove_port_data(port);

	/*
	 * We should just assume the device itself has gone off --
	 * else a close on an open port later will try to send out a
	 * control message.
	 */
	port->portdev = NULL;

	sysfs_remove_group(&port->dev->kobj, &port_attribute_group);
	device_destroy(pdrvdata.class, port->dev->devt);
	cdev_del(port->cdev);

	debugfs_remove(port->debugfs_file);
	kfree(port->name);

	/*
	 * Locks around here are not necessary - a port can't be
	 * opened after we removed the port struct from ports_list
	 * above.
	 */
	kref_put(&port->kref, remove_port);
}

/* Any private messages that the Host and Guest want to share */
static void handle_control_message(struct virtio_device *vdev,
				   struct ports_device *portdev,
				   struct port_buffer *buf)
{
	struct virtio_console_control *cpkt;
	struct port *port;
	size_t name_size;
	int err;
	struct vgpu_device *vgpu;
	u32 nr_gpu;
	size_t prop_size, buf_size;
	int i;
	int start;

	cpkt = (struct virtio_console_control *)(buf->buf + buf->offset);
	// gldebug("cpkt->event = %d\n", cpkt->event);

	if(virtio16_to_cpu(vdev, cpkt->event) == VIRTIO_CONSOLE_VGPU) {
		/*
		 * Skip the size of the header and the cpkt to get the size
		 * of the GPU's prop that was sent
		 */
		buf_size = buf->len - buf->offset - sizeof(*cpkt) ;
		//!!!! be  careful of transferring int type !!!!!!!
		nr_gpu = *(u32*)(buf->buf + buf->offset + sizeof(*cpkt));
		prop_size = (buf_size-sizeof(u32))/nr_gpu;
		start =  buf->offset + sizeof(*cpkt) + sizeof(u32);
		for(i=0; i< nr_gpu; i++) {
			gldebug("one buff size=%lu\n", prop_size);
			vgpu = kmalloc(sizeof(struct vgpu_device), GFP_KERNEL);
			if (!vgpu) {
				dev_err(&portdev->vdev->dev,
					"Not enough space to store vgpu.\n");
				return;
			}
			/* the shared data structure is as follows:
			* struct {
			* 	uint32_t device_id;
			* 	struct cudaDeviceProp prop;
			* }
			*/
			vgpu->id = *(uint32_t*)(buf->buf +start);
			vgpu->flags = 0;
			vgpu->initialized = 0;
			gldebug("vgpu id=%u\n", vgpu->id);
			vgpu->prop_size = prop_size - sizeof(uint32_t);
			vgpu->prop_buf = kmalloc(vgpu->prop_size, GFP_KERNEL);
			if (!vgpu->prop_buf) {
				dev_err(&portdev->vdev->dev,
					"Not enough space to store vgpu prop.\n");
				return;
			}
			memcpy(vgpu->prop_buf, buf->buf +start+sizeof(uint32_t),
					vgpu->prop_size);
			spin_lock_irq(&portdev->vgpus_lock);
			list_add_tail(&vgpu->list, &portdev->vgpus);
			spin_unlock_irq(&portdev->vgpus_lock);
			start+=prop_size;
		}
		portdev->nr_vgpus = nr_gpu;
		// gldebug("portdev->nr_vgpus=%u\n", nr_gpu);
		return;
	}

	port = find_port_by_id(portdev, virtio32_to_cpu(vdev, cpkt->id));
	if (!port &&
	    cpkt->event != cpu_to_virtio16(vdev, VIRTIO_CONSOLE_PORT_ADD)) {
		/* No valid header at start of buffer.  Drop it. */
		dev_dbg(&portdev->vdev->dev,
			"Invalid index %u in control packet\n", cpkt->id);
		return;
	}

	switch (virtio16_to_cpu(vdev, cpkt->event)) {
	case VIRTIO_CONSOLE_PORT_ADD:
		if (port) {
			dev_dbg(&portdev->vdev->dev,
				"Port %u already added\n", port->id);
			send_control_msg(port, VIRTIO_CONSOLE_PORT_READY, 1);
			break;
		}
		if (virtio32_to_cpu(vdev, cpkt->id) >=
		    portdev->max_nr_ports) {
			dev_warn(&portdev->vdev->dev,
				"Request for adding port with "
				"out-of-bound id %u, max. supported id: %u\n",
				cpkt->id, portdev->max_nr_ports - 1);
			break;
		}
		add_port(portdev, virtio32_to_cpu(vdev, cpkt->id));
		break;
	case VIRTIO_CONSOLE_PORT_REMOVE:
		unplug_port(port);
		break;
	case VIRTIO_CONSOLE_CONSOLE_PORT:
		if (!cpkt->value)
			break;
		if (is_console_port(port))
			break;

		init_port_console(port);
		complete(&early_console_added);
		/*
		 * Could remove the port here in case init fails - but
		 * have to notify the host first.
		 */
		break;
	case VIRTIO_CONSOLE_PORT_OPEN:
		port->host_connected = virtio16_to_cpu(vdev, cpkt->value);
		wake_up_interruptible(&port->waitqueue);
		/*
		 * If the host port got closed and the host had any
		 * unconsumed buffers, we'll be able to reclaim them
		 * now.
		 */
		spin_lock_irq(&port->outvq_lock);
		reclaim_consumed_buffers(port);
		spin_unlock_irq(&port->outvq_lock);

		/*
		 * If the guest is connected, it'll be interested in
		 * knowing the host connection state changed.
		 */
		spin_lock_irq(&port->inbuf_lock);
		send_sigio_to_port(port);
		spin_unlock_irq(&port->inbuf_lock);
		break;
	case VIRTIO_CONSOLE_PORT_NAME:
		/*
		 * If we woke up after hibernation, we can get this
		 * again.  Skip it in that case.
		 */
		if (port->name)
			break;

		/*
		 * Skip the size of the header and the cpkt to get the size
		 * of the name that was sent
		 */
		name_size = buf->len - buf->offset - sizeof(*cpkt) + 1;

		port->name = kmalloc(name_size, GFP_KERNEL);
		if (!port->name) {
			dev_err(port->dev,
				"Not enough space to store port name\n");
			break;
		}
		strncpy(port->name, buf->buf + buf->offset + sizeof(*cpkt),
			name_size - 1);
		port->name[name_size - 1] = 0;

		/*
		 * Since we only have one sysfs attribute, 'name',
		 * create it only if we have a name for the port.
		 */
		err = sysfs_create_group(&port->dev->kobj,
					 &port_attribute_group);
		if (err) {
			dev_err(port->dev,
				"Error %d creating sysfs device attributes\n",
				err);
		} else {
			/*
			 * Generate a udev event so that appropriate
			 * symlinks can be created based on udev
			 * rules.
			 */
			kobject_uevent(&port->dev->kobj, KOBJ_CHANGE);
		}
		break;
	}
}

static void control_work_handler(struct work_struct *work)
{
	struct ports_device *portdev;
	struct virtqueue *vq;
	struct port_buffer *buf;
	unsigned int len;

	portdev = container_of(work, struct ports_device, control_work);
	vq = portdev->c_ivq;

	spin_lock(&portdev->c_ivq_lock);
	while ((buf = virtqueue_get_buf(vq, &len))) {
		spin_unlock(&portdev->c_ivq_lock);

		buf->len = len;
		buf->offset = 0;

		handle_control_message(vq->vdev, portdev, buf);

		spin_lock(&portdev->c_ivq_lock);
		if (add_inbuf(portdev->c_ivq, buf) < 0) {
			dev_warn(&portdev->vdev->dev,
				 "Error adding buffer to queue\n");
			free_buf(buf, false);
		}
	}
	spin_unlock(&portdev->c_ivq_lock);
}

static void out_intr(struct virtqueue *vq)
{
	struct port *port;
	func();
	port = find_port_by_vq(vq->vdev->priv, vq);
	if (!port)
		return;

	wake_up_interruptible(&port->waitqueue);
}

static void in_intr(struct virtqueue *vq)
{
	struct port *port;
	unsigned long flags;
	func();
	port = find_port_by_vq(vq->vdev->priv, vq);
	if (!port)
		return;

	spin_lock_irqsave(&port->inbuf_lock, flags);
	port->inbuf = get_inbuf(port);

	/*
	 * Normally the port should not accept data when the port is
	 * closed. For generic serial ports, the host won't (shouldn't)
	 * send data till the guest is connected. But this condition
	 * can be reached when a console port is not yet connected (no
	 * tty is spawned) and the other side sends out data over the
	 * vring, or when a remote devices start sending data before
	 * the ports are opened.
	 *
	 * A generic serial port will discard data if not connected,
	 * while console ports and rproc-serial ports accepts data at
	 * any time. rproc-serial is initiated with guest_connected to
	 * false because port_fops_open expects this. Console ports are
	 * hooked up with an HVC console and is initialized with
	 * guest_connected to true.
	 */

	// if (!port->guest_connected && !is_rproc_serial(port->portdev->vdev))
		discard_port_data(port);

	/* Send a SIGIO indicating new data in case the process asked for it */
	send_sigio_to_port(port);

	spin_unlock_irqrestore(&port->inbuf_lock, flags);

	wake_up_interruptible(&port->waitqueue);
}

static void control_intr(struct virtqueue *vq)
{
	struct ports_device *portdev;

	portdev = vq->vdev->priv;
	schedule_work(&portdev->control_work);
}

static void config_intr(struct virtio_device *vdev)
{
	struct ports_device *portdev;

	portdev = vdev->priv;

	if (!use_multiport(portdev))
		schedule_work(&portdev->config_work);
}

static void config_work_handler(struct work_struct *work)
{
	struct ports_device *portdev;

	portdev = container_of(work, struct ports_device, config_work);
	if (!use_multiport(portdev)) {
		struct virtio_device *vdev;
		struct port *port;
		u16 rows, cols;

		vdev = portdev->vdev;
		virtio_cread(vdev, struct virtio_console_config, cols, &cols);
		virtio_cread(vdev, struct virtio_console_config, rows, &rows);

		port = find_port_by_id(portdev, 0);
	}
}

static int init_vqs(struct ports_device *portdev)
{
	vq_callback_t **io_callbacks;
	char **io_names;
	struct virtqueue **vqs;
	u32 i, j, nr_ports, nr_queues;
	int err;

	nr_ports = portdev->max_nr_ports;
	nr_queues = use_multiport(portdev) ? (nr_ports + 1) * 2 : 2;

	vqs = kmalloc(nr_queues * sizeof(struct virtqueue *), GFP_KERNEL);
	io_callbacks = kmalloc(nr_queues * sizeof(vq_callback_t *), GFP_KERNEL);
	io_names = kmalloc(nr_queues * sizeof(char *), GFP_KERNEL);
	portdev->in_vqs = kmalloc(nr_ports * sizeof(struct virtqueue *),
				  GFP_KERNEL);
	portdev->out_vqs = kmalloc(nr_ports * sizeof(struct virtqueue *),
				   GFP_KERNEL);
	if (!vqs || !io_callbacks || !io_names || !portdev->in_vqs ||
	    !portdev->out_vqs) {
		err = -ENOMEM;
		goto free;
	}

	/*
	 * For backward compat (newer host but older guest), the host
	 * spawns a console port first and also inits the vqs for port
	 * 0 before others.
	 */
	j = 0;
	io_callbacks[j] = in_intr;
	io_callbacks[j + 1] = out_intr;
	io_names[j] = "input";
	io_names[j + 1] = "output";
	j += 2;

	if (use_multiport(portdev)) {
		io_callbacks[j] = control_intr;
		io_callbacks[j + 1] = NULL;
		io_names[j] = "control-i";
		io_names[j + 1] = "control-o";

		for (i = 1; i < nr_ports; i++) {
			j += 2;
			io_callbacks[j] = in_intr;
			io_callbacks[j + 1] = out_intr;
			io_names[j] = "input";
			io_names[j + 1] = "output";
		}
	}
	/* Find the queues. */
	err = virtio_find_vqs(portdev->vdev, nr_queues, vqs,
			      io_callbacks,
			      (const char **)io_names, NULL);
	if (err)
		goto free;

	j = 0;
	portdev->in_vqs[0] = vqs[0];
	portdev->out_vqs[0] = vqs[1];
	j += 2;
	if (use_multiport(portdev)) {
		portdev->c_ivq = vqs[j];
		portdev->c_ovq = vqs[j + 1];

		for (i = 1; i < nr_ports; i++) {
			j += 2;
			portdev->in_vqs[i] = vqs[j];
			portdev->out_vqs[i] = vqs[j + 1];
		}
	}
	kfree(io_names);
	kfree(io_callbacks);
	kfree(vqs);

	return 0;

free:
	kfree(portdev->out_vqs);
	kfree(portdev->in_vqs);
	kfree(io_names);
	kfree(io_callbacks);
	kfree(vqs);

	return err;
}

static const struct file_operations portdev_fops = {
	.owner = THIS_MODULE,
};

static void remove_vqs(struct ports_device *portdev)
{
	portdev->vdev->config->del_vqs(portdev->vdev);
	kfree(portdev->in_vqs);
	kfree(portdev->out_vqs);
}

static void remove_controlq_data(struct ports_device *portdev)
{
	struct port_buffer *buf;
	unsigned int len;

	if (!use_multiport(portdev))
		return;
	
	while ((buf = virtqueue_get_buf(portdev->c_ivq, &len)))
		free_buf(buf, true);

	while ((buf = virtqueue_detach_unused_buf(portdev->c_ivq)))
		free_buf(buf, true);
}

/*
 * Once we're further in boot, we get probed like any other virtio
 * device.
 *
 * If the host also supports multiple console ports, we check the
 * config space to see how many ports the host has spawned.  We
 * initialize each port found.
 */
static int virtcons_probe(struct virtio_device *vdev)
{
	struct ports_device *portdev;
	int err;
	bool multiport;
	bool early = early_put_chars != NULL;

	pr_info("virtio cuda detection!\n");
	/* We only need a config space if features are offered */
	if (!vdev->config->get &&
	    (virtio_has_feature(vdev, VIRTIO_CONSOLE_F_SIZE)
	     || virtio_has_feature(vdev, VIRTIO_CONSOLE_F_MULTIPORT))) {
		dev_err(&vdev->dev, "%s failure: config access disabled\n",
			__func__);
		return -EINVAL;
	}

	/* Ensure to read early_put_chars now */
	barrier();

	portdev = kmalloc(sizeof(*portdev), GFP_KERNEL);
	if (!portdev) {
		err = -ENOMEM;
		goto fail;
	}

	/* Attach this portdev to this virtio_device, and vice-versa. */
	portdev->vdev = vdev;
	vdev->priv = portdev;

	portdev->chr_major = register_chrdev(0, "virtio-portsdev",
					     &portdev_fops);
	gldebug("chr_major = %d\n", portdev->chr_major);
	if (portdev->chr_major < 0) {
		dev_err(&vdev->dev,
			"Error %d registering chrdev for device %u\n",
			portdev->chr_major, vdev->index);
		err = portdev->chr_major;
		goto free;
	}

	multiport = false;
	portdev->max_nr_ports = 1;
	portdev->nr_ports = 0;
	/* Don't test MULTIPORT at all if we're rproc: not a valid feature! */
	if (!is_rproc_serial(vdev) &&
	    virtio_cread_feature(vdev, VIRTIO_CONSOLE_F_MULTIPORT,
				 struct virtio_console_config, max_nr_ports,
				 &portdev->max_nr_ports) == 0) {
		multiport = true;
	}

	err = init_vqs(portdev);
	if (err < 0) {
		dev_err(&vdev->dev, "Error %d initializing vqs\n", err);
		goto free_chrdev;
	}

	spin_lock_init(&portdev->ports_lock);
	INIT_LIST_HEAD(&portdev->ports);
	spin_lock_init(&portdev->vgpus_lock);
	INIT_LIST_HEAD(&portdev->vgpus);

	virtio_device_ready(portdev->vdev);

	INIT_WORK(&portdev->config_work, &config_work_handler);
	INIT_WORK(&portdev->control_work, &control_work_handler);

	gldebug("multiport = %d\n", multiport);
	if (multiport) {
		unsigned int nr_added_bufs;

		spin_lock_init(&portdev->c_ivq_lock);
		spin_lock_init(&portdev->c_ovq_lock);

		nr_added_bufs = fill_queue(portdev->c_ivq,
					   &portdev->c_ivq_lock);
		if (!nr_added_bufs) {
			dev_err(&vdev->dev,
				"Error allocating buffers for control queue\n");
			err = -ENOMEM;
			goto free_vqs;
		}
	} else {
		/*
		 * For backward compatibility: Create a console port
		 * if we're running on older host.
		 */
		add_port(portdev, 0);
	}

	spin_lock_irq(&pdrvdata_lock);
	list_add_tail(&portdev->list, &pdrvdata.portdevs);
	spin_unlock_irq(&pdrvdata_lock);
	gldebug("nr_ports = %d\n", portdev->nr_ports);
	gldebug("max_nr_ports = %d\n", portdev->max_nr_ports);
	proc_file_create(portdev);

	__send_control_msg(portdev, VIRTIO_CONSOLE_BAD_ID,
			   VIRTIO_CONSOLE_DEVICE_READY, 1);
	__send_control_msg(portdev, VIRTIO_CONSOLE_BAD_ID,
			   VIRTIO_CONSOLE_VGPU, 1);

	/*
	 * If there was an early virtio console, assume that there are no
	 * other consoles. We need to wait until the hvc_alloc matches the
	 * hvc_instantiate, otherwise tty_open will complain, resulting in
	 * a "Warning: unable to open an initial console" boot failure.
	 * Without multiport this is done in add_port above. With multiport
	 * this might take some host<->guest communication - thus we have to
	 * wait.
	 */
	if (multiport && early)
		wait_for_completion(&early_console_added);

	return 0;

free_vqs:
	/* The host might want to notify mgmt sw about device add failure */
	__send_control_msg(portdev, VIRTIO_CONSOLE_BAD_ID,
			   VIRTIO_CONSOLE_DEVICE_READY, 0);
	remove_vqs(portdev);
free_chrdev:
	unregister_chrdev(portdev->chr_major, "virtio-portsdev");
free:
	kfree(portdev);
fail:
	return err;
}

static void virtcons_remove(struct virtio_device *vdev)
{
	struct ports_device *portdev;
	struct port *port, *port2;
	struct vgpu_device *vgpu, *vgpu2;
	char name[32];

	
	portdev = vdev->priv;

	spin_lock_irq(&pdrvdata_lock);
	list_del(&portdev->list);
	spin_unlock_irq(&pdrvdata_lock);

	/* Disable interrupts for vqs */
	vdev->config->reset(vdev);
	/* Finish up work that's lined up */
	if (use_multiport(portdev))
		cancel_work_sync(&portdev->control_work);
	else
		cancel_work_sync(&portdev->config_work);

	list_for_each_entry_safe(port, port2, &portdev->ports, list)
		unplug_port(port);

	list_for_each_entry_safe(vgpu, vgpu2, &portdev->vgpus, list) {
		list_del(&vgpu->list);
		kfree(vgpu->prop_buf);
		vgpu->prop_buf = NULL;
		kfree(vgpu);
	}
		


	unregister_chrdev(portdev->chr_major, "virtio-portsdev");

	sprintf(name, "virtual_device_count");
	remove_proc_entry(name, pdrvdata.proc_dir);
	portdev->proc_virt_dev_count = NULL;

	/*
	 * When yanking out a device, we immediately lose the
	 * (device-side) queues.  So there's no point in keeping the
	 * guest side around till we drop our final reference.  This
	 * also means that any ports which are in an open state will
	 * have to just stop using the port, as the vqs are going
	 * away.
	 */
	remove_controlq_data(portdev);
	remove_vqs(portdev);
	kfree(portdev);
}

static struct virtio_device_id id_table[] = {
	{ VIRTIO_ID_CUDA, VIRTIO_DEV_ANY_ID },
	{ 0 },
};

static unsigned int features[] = {
	VIRTIO_CONSOLE_F_SIZE,
	VIRTIO_CONSOLE_F_MULTIPORT,
};


#ifdef CONFIG_PM_SLEEP
static int virtcons_freeze(struct virtio_device *vdev)
{
	struct ports_device *portdev;
	struct port *port;

	portdev = vdev->priv;

	vdev->config->reset(vdev);

	if (use_multiport(portdev))
		virtqueue_disable_cb(portdev->c_ivq);
	cancel_work_sync(&portdev->control_work);
	cancel_work_sync(&portdev->config_work);
	/*
	 * Once more: if control_work_handler() was running, it would
	 * enable the cb as the last step.
	 */
	if (use_multiport(portdev))
		virtqueue_disable_cb(portdev->c_ivq);
	remove_controlq_data(portdev);

	list_for_each_entry(port, &portdev->ports, list) {
		virtqueue_disable_cb(port->in_vq);
		virtqueue_disable_cb(port->out_vq);
		/*
		 * We'll ask the host later if the new invocation has
		 * the port opened or closed.
		 */
		port->host_connected = false;
		remove_port_data(port);
	}
	remove_vqs(portdev);

	return 0;
}

static int virtcons_restore(struct virtio_device *vdev)
{
	struct ports_device *portdev;
	struct port *port;
	int ret;

	portdev = vdev->priv;

	ret = init_vqs(portdev);
	if (ret)
		return ret;

	virtio_device_ready(portdev->vdev);

	if (use_multiport(portdev))
		fill_queue(portdev->c_ivq, &portdev->c_ivq_lock);

	list_for_each_entry(port, &portdev->ports, list) {
		port->in_vq = portdev->in_vqs[port->id];
		port->out_vq = portdev->out_vqs[port->id];

		fill_queue(port->in_vq, &port->inbuf_lock);

		/* Get port open/close status on the host */
		send_control_msg(port, VIRTIO_CONSOLE_PORT_READY, 1);

		/*
		 * If a port was open at the time of suspending, we
		 * have to let the host know that it's still open.
		 */
		if (port->guest_connected)
			send_control_msg(port, VIRTIO_CONSOLE_PORT_OPEN, 1);
	}
	return 0;
}
#endif

static struct virtio_driver virtio_cuda_driver = {
	.feature_table = features,
	.feature_table_size = ARRAY_SIZE(features),
	.driver.name =	KBUILD_MODNAME,
	.driver.owner =	THIS_MODULE,
	.id_table =	id_table,
	.probe =	virtcons_probe,
	.remove =	virtcons_remove,
	.config_changed = config_intr,
#ifdef CONFIG_PM_SLEEP
	.freeze =	virtcons_freeze,
	.restore =	virtcons_restore,
#endif
};


static int my_dev_uevent(struct device *dev, struct kobj_uevent_env *env)
{
	// set char device persmission /dev/<device>
	add_uevent_var(env, "DEVMODE=%#o", 0666);
	return 0;
}

static int __init init(void)
{
	int err;

	pr_info("Hallo virtio cuda!\n");
	pdrvdata.class = class_create(THIS_MODULE, "virtio-cuda");
	if (IS_ERR(pdrvdata.class)) {
		err = PTR_ERR(pdrvdata.class);
		pr_err("Error %d creating virtio-cuda class\n", err);
		return err;
	}
	pdrvdata.class->dev_uevent = my_dev_uevent;

	pdrvdata.debugfs_dir = debugfs_create_dir("virtio-cuda", NULL);
	if (!pdrvdata.debugfs_dir)
		pr_warn("Error creating debugfs dir for virtio-cuda\n");
	pdrvdata.proc_dir = proc_mkdir("virtio-cuda", NULL);
	if (!pdrvdata.proc_dir)
		pr_warn("Error creating proc dir for /proc/virtio-cuda\n");
	INIT_LIST_HEAD(&pdrvdata.consoles);
	INIT_LIST_HEAD(&pdrvdata.portdevs);

	err = register_virtio_driver(&virtio_cuda_driver);
	if (err < 0) {
		pr_err("Error %d registering virtio driver\n", err);
		goto free;
	}

	return 0;

free:
	debugfs_remove_recursive(pdrvdata.debugfs_dir);
	proc_remove(pdrvdata.proc_dir);
	class_destroy(pdrvdata.class);
	return err;
}

static void __exit fini(void)
{
	reclaim_dma_bufs();

	unregister_virtio_driver(&virtio_cuda_driver);

	class_destroy(pdrvdata.class);
	debugfs_remove_recursive(pdrvdata.debugfs_dir);
	proc_remove(pdrvdata.proc_dir);
	pr_info("Bye bye virtio cuda!\n");
}
module_init(init);
module_exit(fini);

MODULE_DEVICE_TABLE(virtio, id_table);
MODULE_DESCRIPTION("Virtio cuda driver");
MODULE_LICENSE("GPL");
