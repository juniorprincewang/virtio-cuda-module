#ifndef _LINUX_LIST_H
#define _LINUX_LIST_H

#ifdef __cplusplus
extern "C" {
#endif

/* a list structure: we could use Linux's list_head, but it's not available
   in user-space - hence use our own list structure. */
struct list_head {
    struct list_head *next;
    struct list_head *prev;
	void *container;
};

static inline void list_head_init(struct list_head *entry, void *container)
{
	entry->next = entry->prev = entry; /* used to be "= NULL" */
	entry->container = container;
}

static inline void list_add_next(struct list_head *entry, struct list_head *pos)
{
	struct list_head *next = pos->next;

	entry->next = next;
	next->prev = entry;
	entry->prev = pos;
	pos->next = entry;
}

static inline void list_add_prev(struct list_head *entry, struct list_head *pos)
{
	struct list_head *prev = pos->prev;

	entry->prev = prev;
	prev->next = entry;
	entry->next = pos;
	pos->prev = entry;
}

static inline void list_add(struct list_head *entry, struct list_head *head)
{
	return list_add_next(entry, head);
}

static inline void list_add_tail(struct list_head *entry, struct list_head *head)
{
	return list_add_prev(entry, head);
}

static inline void list_del(struct list_head *entry)
{
	struct list_head *next = entry->next;
	struct list_head *prev = entry->prev;

	/* if prev is null, @entry points to the head, hence something wrong. */
	prev->next = next;
	next->prev = prev;
	entry->next = entry->prev = entry;
}

static inline int list_empty(struct list_head *entry)
{
	return (entry->next == entry->prev) && (entry->next == entry);
}

static inline struct list_head *__list_head(struct list_head *head)
{
	/* head->next is the actual head of the list. */
	return (head && !list_empty(head)) ? head->next : NULL;
}

static inline void *list_container(struct list_head *entry)
{
	return entry ? entry->container : NULL;
}

#define list_for_each_entry(p, list, entry_name)	\
	for (p = list_container(__list_head(list));		\
		 p != NULL;									\
		 p = list_container((p)->entry_name.next))



#define list_for_each_entry_type(TYPE, p, list, entry_name)	\
	for (p = (TYPE *)list_container(__list_head(list));		\
		 p != NULL;									\
		 p = (TYPE *)list_container((p)->entry_name.next))



#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif