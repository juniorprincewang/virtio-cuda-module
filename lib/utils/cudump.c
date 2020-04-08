#include "cudump.h"


static void cubin_func_skip(char **pos, section_entry_t *e)
{
	*pos += sizeof(section_entry_t);
#ifdef CUDUMP_DEBUG
	printf("/* nv.info: ignore entry type: 0x%04x, size=0x%x */\n",
		   e->type, e->size);
	if (e->size % 4 == 0) {
		int i;
		for (i = 0; i < e->size / 4; i++) {
			uint32_t val = ((uint32_t*)*pos)[i];
			printf("0x%04x\n", val);
		}
	}
	else {
		int i;
		for (i = 0; i < e->size; i++) {
			unsigned char val = ((unsigned char*)*pos)[i];
			printf("0x%02x\n", (uint32_t)val);
		}
	}
#endif
	*pos += e->size;
}

static void cubin_func_unknown(char **pos, section_entry_t *e)
{
	printf("/* nv.info: unknown entry type: 0x%.4x, size=0x%x */\n",
			   e->type, e->size);
	cubin_func_skip(pos, e);
}

static int cubin_func_0a04
(char **pos, section_entry_t *e, struct cuda_raw_func *raw_func)
{
	const_entry_t *ce;

	*pos += sizeof(section_entry_t);
	ce = (const_entry_t *)*pos;
	raw_func->param_base = ce->base;
	raw_func->param_size = ce->size;
	*pos += e->size;

	return 0;
}

static int cubin_func_0c04
(char **pos, section_entry_t *e, struct cuda_raw_func *raw_func)
{
	*pos += sizeof(section_entry_t);
	/* e->size is a parameter size, but how can we use it here? */
	*pos += e->size;

	return 0;
}

static int cubin_func_0d04
(char **pos, section_entry_t *e, struct cuda_raw_func *raw_func)
{
	stack_entry_t *se;

	*pos += sizeof(section_entry_t);
	se = (stack_entry_t*) *pos;
	raw_func->stack_depth = se->size;
	/* what is se->unk16 and se->unk32... */

	*pos += e->size;

	return 0;
}

static int cubin_func_1704
(char **pos, section_entry_t *e, struct cuda_raw_func *raw_func)
{
	param_entry_t *pe;
	struct cuda_param *param_data;

	*pos += sizeof(section_entry_t);
	pe = (param_entry_t *)*pos;

	param_data = (struct cuda_param *)malloc(sizeof(*param_data));
	param_data->idx = pe->idx;
	param_data->offset = pe->offset;
	param_data->size = pe->size >> 18;
	param_data->flags = pe->size & 0x2ffff;
	
	/* append to the head of the parameter data list. */
	param_data->next = raw_func->param_data;
	raw_func->param_data = param_data;

	*pos += e->size;

	return 0;
}

static int cubin_func_1903
(char **pos, section_entry_t *e, struct cuda_raw_func *raw_func)
{
	int ret;
	char *pos2;

	*pos += sizeof(section_entry_t);
	pos2 = *pos;

	/* obtain parameters information. is this really safe? */
	do {
		section_entry_t *sh_e = (section_entry_t *)pos2;
		ret = cubin_func_1704(&pos2, sh_e, raw_func);
		if (ret)
			return ret;
		raw_func->param_count++;
	} while (((section_entry_t *)pos2)->type == 0x1704);

	/* just check if the parameter size matches. */
	if (raw_func->param_size != e->size) {
		if (e->type == 0x1803) { /* sm_13 needs to set param_size here. */
			raw_func->param_size = e->size;
		}
		else {
			printf("Parameter size mismatched\n");
			printf("0x%x and 0x%x\n", raw_func->param_size, e->size);
		}
	}

	*pos = pos2; /* need to check if this is correct! */

	return 0;
}

static int cubin_func_1e04
(char **pos, section_entry_t *e, struct cuda_raw_func *raw_func)
{
	crs_stack_size_entry_t *crse;

	*pos += sizeof(section_entry_t);
	crse = (crs_stack_size_entry_t*) *pos;
	raw_func->stack_size = crse->size << 4;

	*pos += e->size;

	return 0;
}

static int cubin_func_type
(char **pos, section_entry_t *e, struct cuda_raw_func *raw_func)
{
	switch (e->type) {
	case 0x0204: /* textures */
		cubin_func_skip(pos, e);
		break;
	case 0x0a04: /* kernel parameters base and size */
		return cubin_func_0a04(pos, e, raw_func);
	case 0x0b04: /* 4-byte align data relevant to params (sm_13) */
	case 0x0c04: /* 4-byte align data relevant to params (sm_20) */
		return cubin_func_0c04(pos, e, raw_func);
	case 0x0d04: /* stack information, hmm... */
		return cubin_func_0d04(pos, e, raw_func);
	case 0x1104: /* ignore recursive call */
		cubin_func_skip(pos, e);
		break;
	case 0x1204: /* some counters but what is this? */
		cubin_func_skip(pos, e);
		break;
	case 0x1803: /* kernel parameters itself (sm_13) */
	case 0x1903: /* kernel parameters itself (sm_20/sm_30) */
		return cubin_func_1903(pos, e, raw_func);
	case 0x1704: /* each parameter information */
		return cubin_func_1704(pos, e, raw_func);
	case 0x1e04: /* crs stack size information */
		return cubin_func_1e04(pos, e, raw_func);
	case 0x0001: /* ??? */
		cubin_func_skip(pos, e);
		break;
	case 0x1b03: /*sm 35 unknow*/
		cubin_func_skip(pos, e);
		break;
	case 0x080d: /* ??? */
		cubin_func_skip(pos, e);
		break;
	case 0xf000: /* maybe just padding??? */
		*pos += 4;
		break;
	case 0xffff: /* ??? */
		cubin_func_skip(pos, e);
		break;
	case 0x0020: /* ??? */
		cubin_func_skip(pos, e);
		break;
	default: /* real unknown */
		cubin_func_unknown(pos, e);
		/* return -EINVAL; */
	}

	return 0;
}

static void destroy_all_symbols(struct CUmod_st *mod)
{
	struct cuda_const_symbol *cs;
	struct list_head *p, *head;

	head = &mod->symbol_list;
	while(!list_empty(head)) {
		p = head->next;
		list_del(p);
		cs = list_container(p);
		free(cs);
	}
}


static void destroy_all_functions(struct CUmod_st *mod)
{
/*	struct CUfunc_st *func, *func2;
	struct cuda_raw_func *raw_func;
	struct cuda_param *param_data;
	struct list_head *p;
	list_for_each_entry_safe(func, func2, &mod->func_list, list_entry) {
		list_del(&func->list_entry);
		raw_func = &func->raw_func;
		while (raw_func->param_data) {
			param_data = raw_func->param_data;
			raw_func->param_data = raw_func->param_data->next;
			free(param_data);
		}
		free(raw_func->name);
		free(func);
	}*/
}


struct CUfunc_st* lookup_func_by_name(struct CUmod_st *mod, const char *name)
{
	struct CUfunc_st *func;
	list_for_each_entry(func, &mod->func_list, list_entry) {
		if (strcmp(func->raw_func.name, name) == 0) {
			return func;
		}
	}
	return NULL;
}

struct CUfunc_st* lookup_func_by_hostfunc(struct CUmod_st *mod, const void *host_func)
{
	struct CUfunc_st *func;
	list_for_each_entry(func, &mod->func_list, list_entry) {
		if (func->raw_func.host_func == host_func) {
			return func;
		}
	}
	return NULL;
}

struct cuda_const_symbol* lookup_symbol_by_name(struct CUmod_st *mod, const char *name)
{
	struct cuda_const_symbol *cs;
	list_for_each_entry(cs, &mod->symbol_list, list_entry) {
		if (strcmp(cs->name, name) == 0) {
			return cs;
		}
	}
	return NULL;
}

/* global/constant variable dump. */
void dump_symbol(struct CUmod_st *mod)
{
    struct cuda_const_symbol *cs;
    printf("constant symbols: \n");

    list_for_each_entry(cs, &mod->symbol_list, list_entry) {
        printf("idx %d, name %s, offset %d, size %d\n", 
            cs->idx, cs->name, cs->offset, cs->size);
    }
    printf("\n");
}

/* dump kernel function and parameters */
void dump_kernel(struct CUmod_st *mod)
{
    struct CUfunc_st *func = NULL;
    struct cuda_raw_func *f = NULL;
    struct cuda_param *param_data = NULL;
    printf("kernel symbols: \n");

    list_for_each_entry(func, &mod->func_list, list_entry) {
        f = &func->raw_func;
        printf("func name %s, size %d\n", 
            f->name, f->code_size);
        printf("func code buf %p\n", f->code_buf);
        printf("\t},\n");
        printf("\t.param_base = 0x%x,\n", f->param_base);
        printf("\t.param_size = 0x%x,\n", f->param_size);
        printf("\t.param_count = 0x%x,\n", f->param_count);
        printf("\t.param_data = {\n");
        param_data = f->param_data;
        while (param_data) {
            printf("\t\t{%d, 0x%x, 0x%x, 0x%x},\n", 
                   param_data->idx, 
                   param_data->offset, 
                   param_data->size, 
                   param_data->flags);
            param_data = param_data->next;
        }
    }
    printf("\n");
}

static void init_mod(struct CUmod_st *mod, char *bin)
{
	int i;

	mod->bin = bin;
	mod->func_count = 0;
	mod->symbol_count = 0;
	for (i = 0; i < NVIDIA_CONST_SEGMENT_MAX_COUNT; i++) {
		mod->cmem[i].addr = 0;
		mod->cmem[i].size = 0;
		mod->cmem[i].raw_size = 0;
		mod->cmem[i].buf = NULL;
	}
	list_head_init(&mod->func_list, NULL);
	list_head_init(&mod->symbol_list, NULL);
	mod->arch = 0;
}

static void init_raw_func(struct cuda_raw_func *f)
{
	int i;

	f->name = NULL;
	f->code_buf = NULL;
	f->code_size = 0;
	for (i = 0; i < NVIDIA_CONST_SEGMENT_MAX_COUNT; i++) {
		f->cmem[i].buf = NULL;
		f->cmem[i].size = 0;
	}
	f->reg_count = 0;
	f->bar_count = 0;
	f->stack_depth = 0;
	f->stack_size = 0;
	f->shared_size = 0;
	f->param_base = 0;
	f->param_size = 0;
	f->param_count = 0;
	f->param_data = NULL;
	f->local_size = 0;
	f->local_size_neg = 0;
}

static struct CUfunc_st* malloc_func_if_necessary(struct CUmod_st *mod, const char *name)
{
	struct CUfunc_st *func = NULL;
	if ((func = lookup_func_by_name(mod, name))) {
		return func;
	}

	/* We allocate and initialize func and link it to mod's linked list. */
	func = malloc(sizeof(*func));
	if (!func) {
		return NULL;
	}
	init_raw_func(&func->raw_func);
	func->raw_func.name = strdup(name);

	/* insert this function to the module's function list. */
	list_head_init(&func->list_entry, func);
	list_add(&func->list_entry, &mod->func_list);
	mod->func_count++;
	func->mod = mod;
	return func;
}

static int load_cubin(struct CUmod_st *mod, char *bin)
{
	Elf_Ehdr *ehead;
	Elf_Shdr *sheads;
	Elf_Phdr *pheads;
	Elf_Sym *symbols, *sym;
	char *strings;
	char *shstrings;
	char *nvinfo, *nvrel, *nvglobal_init;
	uint32_t symbols_size;
	int symbols_idx, strings_idx;
	int nvinfo_idx, nvrel_idx, nvrel_const_idx,	nvglobal_idx, nvglobal_init_idx;
	symbol_entry_t *sym_entry;
	section_entry_t *se;
	void *sh;
	char *sh_name;
	char *pos;
	int i, ret = 0;

	if (memcmp(bin, "\177ELF", 4))
		return -ENOENT;

	/* initialize ELF variables. */
	ehead = (Elf_Ehdr *)bin;
	sheads = (Elf_Shdr *)(bin + ehead->e_shoff);
	pheads = (Elf_Phdr *)(bin + ehead->e_phoff);
	symbols = NULL;
	strings = NULL;
	nvinfo = NULL;
	nvrel = NULL;
	nvglobal_init = NULL;
	symbols_idx = 0;
	strings_idx = 0;
	nvinfo_idx = 0;
	nvrel_idx = 0;
	nvrel_const_idx = 0;
	nvglobal_idx = 0;
	nvglobal_init_idx = 0;
	shstrings = bin + sheads[ehead->e_shstrndx].sh_offset;

	/* seek the ELF header. */
	for (i = 0; i < ehead->e_shnum; i++) {
		sh_name = (char *)(shstrings + sheads[i].sh_name);
		sh = bin + sheads[i].sh_offset;
		/* the following are function-independent sections. */
		switch (sheads[i].sh_type) {
		case SHT_SYMTAB: /* symbol table */
			symbols_idx = i;
			symbols = (Elf_Sym *)sh;
			break;
		case SHT_STRTAB: /* string table */
			strings_idx = i;
			strings = (char *)sh;
			break;
		case SHT_REL: /* relocatable: not sure if nvcc uses it... */
			nvrel_idx = i;
			nvrel = (char *)sh;
			sscanf(sh_name, "%*s%d", &nvrel_const_idx);
			break;
		default:
			/* we never know what sections (.text.XXX, .info.XXX, etc.)
			   appears first for each function XXX... */
			if (!strncmp(sh_name, SH_TEXT, strlen(SH_TEXT))) {
				struct CUfunc_st *func = NULL;
				struct cuda_raw_func *raw_func = NULL;

				/* this function does nothing if func is already allocated. */
				func = malloc_func_if_necessary(mod, sh_name + strlen(SH_TEXT));
				if (!func)
					goto fail_malloc_func;

				raw_func = &func->raw_func;

				/* basic information. */
				raw_func->code_buf = bin + sheads[i].sh_offset; /* ==sh */
				raw_func->code_size = sheads[i].sh_size;
				raw_func->reg_count = (sheads[i].sh_info >> 24) & 0x3f;
				raw_func->bar_count = (sheads[i].sh_flags >> 20) & 0xf;
			}
			else if (!strncmp(sh_name, SH_CONST, strlen(SH_CONST))) {
				char fname[256] = {0};
				int x; /* cX[] */
				sscanf(sh_name, SH_CONST "%d.%s", &x, fname);
				/* global constant spaces. */
				if (strlen(fname) == 0) {
					mod->cmem[x].buf = bin + sheads[i].sh_offset;
					mod->cmem[x].raw_size = sheads[i].sh_size;
				}
				else if (x >= 0 && x < NVIDIA_CONST_SEGMENT_MAX_COUNT) {
					struct CUfunc_st *func = NULL;
					/* this function does nothing if func is already allocated. */
					func = malloc_func_if_necessary(mod, fname);
					if (!func)
						goto fail_malloc_func;
					func->raw_func.cmem[x].buf = bin + sheads[i].sh_offset;
					func->raw_func.cmem[x].size = sheads[i].sh_size;
				}
			}
			else if (!strncmp(sh_name, SH_SHARED, strlen(SH_SHARED))) {
				struct CUfunc_st *func = NULL;
				/* this function does nothing if func is already allocated. */
				func =  malloc_func_if_necessary(mod, sh_name + strlen(SH_SHARED));
				if (!func)
					goto fail_malloc_func;
				func->raw_func.shared_size = sheads[i].sh_size;
				/*
				 * int x;
				 * for (x = 0; x < raw_func->shared_size/4; x++) {
				 * 		unsigned long *data = bin + sheads[i].sh_offset;
				 *		printf("0x%x: 0x%x\n", x*4, data[x]);
				 * }
				 */
			}
			else if (!strncmp(sh_name, SH_LOCAL, strlen(SH_LOCAL))) {
				struct CUfunc_st *func = NULL;
				/* this function does nothing if func is already allocated. */
				func = malloc_func_if_necessary(mod, sh_name + strlen(SH_LOCAL));
				if (!func)
					goto fail_malloc_func;
				func->raw_func.local_size = sheads[i].sh_size;
				func->raw_func.local_size_neg = 0x7c0; /* FIXME */
			}
			/* NOTE: there are two types of "info" sections: 
			   1. ".nv.info.funcname"
			   2. ".nv.info"
			   ".nv.info.funcname" represents function information while 
			   ".nv.info" points to all ".nv.info.funcname" sections and
			   provide some global data information.
			   NV50 doesn't support ".nv.info" section. 
			   we also assume that ".nv.info.funcname" is an end mark. */
			else if (!strncmp(sh_name, SH_INFO_FUNC, strlen(SH_INFO_FUNC))) {
				struct CUfunc_st *func = NULL;
				struct cuda_raw_func *raw_func = NULL;
				/* this function does nothing if func is already allocated. */
				func = malloc_func_if_necessary(mod, sh_name + strlen(SH_INFO_FUNC));
				if (!func)
					goto fail_malloc_func;

				raw_func = &func->raw_func;

				/* look into the nv.info.@raw_func->name information. */
				pos = (char *) sh;
				while (pos < (char *) sh + sheads[i].sh_size) {
					se = (section_entry_t*) pos;
					ret = cubin_func_type(&pos, se, raw_func);
					if (ret)
						goto fail_cubin_func_type;
				}
			}
			else if (!strcmp(sh_name, SH_INFO)) {
				nvinfo_idx = i;
				nvinfo = (char *) sh;
			}
			else if (!strcmp(sh_name, SH_GLOBAL)) {
				/* symbol space size. */
				symbols_size = sheads[i].sh_size;
				nvglobal_idx = i;
			}
			else if (!strcmp(sh_name, SH_GLOBAL_INIT)) {
				nvglobal_init_idx = i;
				nvglobal_init = (char *) sh;
			}
			break;
		}
	}

	/* nv.rel... "__device__" symbols? */
	for (sym_entry = (symbol_entry_t *)nvrel; 
		 (void *)sym_entry < (void *)nvrel + sheads[nvrel_idx].sh_size;
		 sym_entry++) {
		/*
		 char *sym_name, *sh_name;
		 uint32_t size;
		 sym  = &symbols[se->sym_idx];
		 sym_name = strings + sym->st_name;
		 sh_name = strings + sheads[sym->st_shndx].sh_name;
		 size = sym->st_size;
		*/
	}

	/* symbols: __constant__ variable and built-in function names. */
	for (sym = &symbols[0]; 
		 (void *)sym < (void *)symbols + sheads[symbols_idx].sh_size; sym++) {
		 char *sym_name = strings + sym->st_name;
		 char *sh_name = shstrings + sheads[sym->st_shndx].sh_name;
		 switch (sym->st_info) {
		 case 0x0: /* ??? */
			 break;
		 case 0x2: /* ??? */
			 break;
		 case 0x3: /* ??? */
			 break;
		 case 0x1:
		 case 0x11: /* __device__/__constant__ symbols */
			 if (sym->st_shndx == nvglobal_idx) { /* __device__ */
			 }
			 else { /* __constant__ */
				 int x;
				 struct cuda_const_symbol *cs = malloc(sizeof(*cs));
				 if (!cs) {
					 ret = -ENOMEM;
					 goto fail_symbol;
				 }
				 sscanf(sh_name, SH_CONST"%d", &x);
				 cs->idx = x;
				 cs->name = sym_name;
				 cs->offset = sym->st_value;
				 cs->size = sym->st_size;
				 list_head_init(&cs->list_entry, cs);
				 list_add(&cs->list_entry, &mod->symbol_list);
				 mod->symbol_count++;
			 }
			 break;
		 case 0x12: /* function symbols */
			 break;
		 case 0x22: /* quick hack: FIXME! */
			 printf("sym_name: %s\n", sym_name);
			 printf("sh_name: %s\n", sh_name);
			 printf("st_value: 0x%llx\n", (unsigned long long)sym->st_value);
			 printf("st_size: 0x%llx\n", (unsigned long long)sym->st_size);
			 break;
		 default: /* ??? */
			 printf("/* unknown symbols: 0x%x\n */", sym->st_info);
			 goto fail_symbol;
		 }
	}
	if (nvinfo) { /* >= sm_20 */
		/* parse nv.info sections. */
		pos = (char*)nvinfo;
		while (pos < nvinfo + sheads[nvinfo_idx].sh_size) {
			section_entry_t *e = (section_entry_t*) pos;
			switch (e->type) {
			case 0x0704: /* texture */
				cubin_func_skip(&pos, e);
				break;
			case 0x1104:  /* function */
				cubin_func_skip(&pos, e);
				break;
			case 0x1204: /* some counters but what is this? */
				cubin_func_skip(&pos, e);
				break;
			case 0x2304: /*sm 35 unknow*/
				cubin_func_skip(&pos, e);
				break;
			default:
				cubin_func_unknown(&pos, e);
				/* goto fail_function; */
			}
		}
		mod->arch = CUDA_ARCH_SM_2X;
	}
	else { /* < sm_13 */
		mod->arch = CUDA_ARCH_SM_1X;
	}

	return 0;

fail_symbol:
fail_cubin_func_type:
fail_malloc_func:
	destroy_all_functions(mod);

	return ret;
}

static void unload_cubin(struct CUmod_st *mod)
{
	if (mod->bin) {
		free(mod->bin);
		mod->bin = NULL;
	}
}

static int load_file(char **pbin, const char *fname)
{
	char *bin;
	FILE *fp;
	uint32_t len;

	if (!(fp = fopen(fname, "rb")))
		return -ENOENT;

	fseek(fp, 0, SEEK_END);
	len = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	if (!(bin = (char *) malloc(len + 1))) {
		fclose(fp);
		return -ENOMEM;
	}

	if (!fread(bin, len, 1, fp)) {
		free(bin);
		fclose(fp);
		return -EIO;
	}
	fclose(fp);
	*pbin = bin;

	return 0;
}

int cuda_load_cubin(struct CUmod_st *mod, const char *bin)
{
	/* initialize module. */
	init_mod(mod, bin);

	if (load_cubin(mod, bin)) {
		fprintf(stderr, "Failed to load cubin\n");
		unload_cubin(mod);
		return 1;
	}
	return 0;
}

int cuda_load_cubin_file(struct CUmod_st *mod, const char *fname)
{
	char *bin;
	int ret;

	ret = load_file(&bin, fname);
	if (ret) {
		fprintf(stderr, "Failed to load file %s\n", fname);
		return 1;
	}

	/* initialize module. */
	init_mod(mod, bin);

	ret = load_cubin(mod, bin);
	if (ret) {
		fprintf(stderr, "Failed to load cubin\n");
		unload_cubin(mod);
		return 1;
	}

	return 0;
}
