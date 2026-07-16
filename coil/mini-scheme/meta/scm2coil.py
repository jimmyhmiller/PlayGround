#!/usr/bin/env python3
# Scheme -> Coil transpiler frontend. Maps the first-order Scheme subset onto the
# Val runtime (sval.coil) so the transparent-GC metaprogram (gcauto2.coil) can
# compile it to native code. Reads a .scm on argv[1], writes a .coil to stdout.
import sys
def tokenize(s):
    out=[]; i=0
    while i<len(s):
        c=s[i]
        if c==';':
            while i<len(s) and s[i]!='\n': i+=1
        elif c in ' \t\n': i+=1
        elif c in '()': out.append(c); i+=1
        else:
            j=i
            while j<len(s) and s[j] not in ' \t\n()': j+=1
            out.append(s[i:j]); i=j
    return out
def parse(toks):
    forms=[]
    def rd(it):
        t=next(it)
        if t=='(':
            l=[]
            while True:
                t2=next(it2peek(it))
                if t2==')': next(it); return l
                l.append(rd(it))
        return atom(t)
    it=iter(toks)
    import itertools
    # simple recursive parser
    pos=[0]
    def peek(): return toks[pos[0]] if pos[0]<len(toks) else None
    def nxt(): v=toks[pos[0]]; pos[0]+=1; return v
    def rd2():
        t=nxt()
        if t=='(':
            l=[]
            while peek()!=')': l.append(rd2())
            nxt(); return l
        return atom(t)
    while pos[0]<len(toks): forms.append(rd2())
    return forms
def it2peek(it): return it
def atom(t):
    try: return ('int', int(t))
    except: return ('sym', t)

PRIMS={'car':'car','cdr':'cdr','cons':'cons','eq?':'s-eq','null?':'s-null',
       'number?':'s-num','symbol?':'s-symp','pair?':'s-pair','+':'s-add','-':'s-sub',
       '*':'s-mul','<':'s-lt','=':'s-eqn','not':'s-not'}

def is_atom(x): return isinstance(x,tuple)
def head(x): return x[0][1] if (isinstance(x,list) and x and is_atom(x[0]) and x[0][0]=='sym') else None

# symbols compared against a (quote sym) literal — we cache their interned id once
# and compare by i64 instead of re-interning + re-boxing on every dispatch.
QSYMS={}          # name -> sanitized ident
_SAN={'+':'plus','-':'minus','*':'star','/':'slash','<':'lt','>':'gt','=':'eq',
      '<=':'le','>=':'ge','?':'q','!':'bang'}
def sanitize(name):
    return ''.join(_SAN.get(c,c) if not (c.isalnum()) else c for c in name) or 'sym'
def qid_of(name):
    san=QSYMS.get(name)
    if san is None:
        base='qid-'+sanitize(name); san=base; k=2
        while san in QSYMS.values(): san=base+str(k); k+=1
        QSYMS[name]=san
    return f'({QSYMS[name]})'
def quoted_sym(x):        # (quote <symbol>) -> its name, else None
    if isinstance(x,list) and len(x)==2 and head(x)=='quote' \
       and is_atom(x[1]) and x[1][0]=='sym': return x[1][1]
    return None

def cquote(x):
    if is_atom(x):
        if x[0]=='int': return f'(mk-int {x[1]})'
        return f'(mk-sym (intern "{x[1]}"))'
    # list -> nested cons ending in snil
    r='(snil)'
    for e in reversed(x): r=f'(cons {cquote(e)} {r})'
    return r

# If `x` is `(eq? A (quote sym))` (either order), return a raw-bool comparison by
# interned id — no re-intern, no boxing. Used only in condition position, where the
# surrounding `if` wants a bool directly (not a Val passed through `truthy`).
def bool_cond(x, locals, gvals):
    if head(x)=='eq?' and isinstance(x,list) and len(x)==3:
        for a,b in ((x[1],x[2]),(x[2],x[1])):
            s=quoted_sym(b)
            if s is not None:
                return f'(= (hid {cexpr(a,locals,gvals)}) {qid_of(s)})'
    return None

def cexpr(x, locals, gvals):
    if is_atom(x):
        if x[0]=='int': return f'(mk-int {x[1]})'
        s=x[1]
        if s in locals: return s
        if s=='genv': return '(genv)'
        if s in gvals: return f'({s})'
        return s  # fallback: treat as local
    h=head(x)
    if h=='quote': return cquote(x[1])
    if h=='if':
        cond=x[1]; bc=bool_cond(cond,locals,gvals)
        c=bc if bc is not None else f'(truthy {cexpr(cond,locals,gvals)})'
        t=cexpr(x[2],locals,gvals); e=cexpr(x[3],locals,gvals)
        return f'(if {c} {t} {e})'
    if h=='begin':
        return '(do '+' '.join(cexpr(e,locals,gvals) for e in x[1:])+')'
    if h=='set!':
        # only set! on genv
        return f'(set-genv! {cexpr(x[2],locals,gvals)})'
    if h in PRIMS:
        args=' '.join(cexpr(a,locals,gvals) for a in x[1:])
        return f'({PRIMS[h]} {args})'
    # user function call
    args=' '.join(cexpr(a,locals,gvals) for a in x[1:])
    return f'({h} {args})' if not x[1:] else f'({h} {args})'

def main():
    src=open(sys.argv[1]).read()
    forms=parse(tokenize(src))
    # classify
    fns=[]; gvals=set(); toplevel=[]
    defs=[]
    for f in forms:
        if head(f)=='define':
            sig=f[1]
            if isinstance(sig,list):  # function
                fns.append(f)
            else:  # value global
                name=sig[1]
                if name!='genv': gvals.add(name)
                defs.append(('val',f))
        else:
            toplevel.append(f)
    # gather all fn + gval names for reference resolution
    for f in fns:
        pass
    body_out=[]
    # value globals as zero-arg fns (except genv)
    for kind,f in defs:
        name=f[1][1]
        if name=='genv': continue
        body=cexpr(f[2], set(), gvals)
        body_out.append(f'(defn {name} [] (-> Val) {body})')
    # functions
    for f in fns:
        sig=f[1]; name=sig[0][1]; params=[p[1] for p in sig[1:]]
        locs=set(params)
        pdecl=' '.join(f'({p} Val)' for p in params)
        bodies=[cexpr(b,locs,gvals) for b in f[2:]]
        body=bodies[0] if len(bodies)==1 else '(do '+' '.join(bodies)+')'
        body_out.append(f'(defn {name} [{pdecl}] (-> Val) {body})')

    out=[]
    out.append('(module scheme)')
    out.append('(import "../sval.coil" :use *)')
    out.append('(import "../gcauto2.coil" :use *)   ; the GC metaprogram')
    out.append('(import "io.coil" :use *)')
    out.append('(import "fmt.coil" :use *)')
    out.append('')
    # cached special-form ids: intern ONCE (lazy, +1 sentinel so 0 == uninit),
    # then compare (car e) by i64 instead of re-interning + re-boxing per dispatch.
    out.append('(defn hid [(v Val)] (-> i64) (if (symp v) (sym-id v) -1))')
    for name,san in QSYMS.items():
        out.append(f'(defn {san}-cell [] (-> (ptr i64)) (alloc-static i64))')
        out.append(f'(defn {san} [] (-> i64) (let [c ({san}-cell)] '
                   f'(do (if (= (load c) 0) (store! c (+ (intern "{name}") 1)) 0) (- (load c) 1))))')
    out.append('')
    out.extend(body_out)
    # main: init genv, run top-level display forms
    mainbody=['(set-genv! (snil))']
    for t in toplevel:
        h=head(t)
        if h=='display': mainbody.append(f'(print-val {cexpr(t[1],set(),gvals)})')
        elif h=='newline': mainbody.append('(fmt (stdout) "\\n")')
    mainbody.append('(fmt (stderr) "[gc] peak_live={d}  collections={d}  total_alloc={d}\\n" (gc-peak) (gc-collections) (gc-total))'); mainbody.append('0')
    out.append('(defn main [] (-> :i64) (do '+' '.join(mainbody)+'))')
    print('\n'.join(out))
main()
