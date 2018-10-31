import itertools

import fdb
import fdb.tuple

from pyDatalog import pyDatalog, pyEngine

fdb.api_version(300)

db = fdb.open()

pyDatalog.create_atoms('kvp')


class Datalog(object):

    def __init__(self, subspace, max_arity):
        self.subspace = subspace
        self.max_arity = max_arity
        self._create_generic_fdb()

    #####################################
    ##   Generic FDB Predicates: kvp   ##
    #####################################

    @fdb.transactional
    def _get_generic_predicate(self, tr, prefix_tuple, partial_key):
        if partial_key:
            for k, v in tr[self.subspace.range(prefix_tuple)]:
                yield k, v
        else:
            k = self.subspace.pack(prefix_tuple)
            v = tr[k]
            if v.present():
                yield k, v

    def _resolve_generic_fdb(self, arity):

        def func(*args):
            assert len(args) == arity, "arity mismatch"
            leftmost_consts = [arg.id for arg in
                               itertools.takewhile(lambda x: x.is_const(), args[:-1])]
            prefix_tuple = tuple(leftmost_consts)
            partial_key = len(leftmost_consts) < arity - 1
            for k, v in self._get_generic_predicate(db, prefix_tuple, partial_key):
                yield self.subspace.unpack(k)+(v,)

        str_arity = str(arity)
        pyEngine.Python_resolvers['kvp'+str_arity+'/'+str_arity] = func

    def _create_generic_fdb(self):
        for arity in range(1, self.max_arity+1):
            self._resolve_generic_fdb(arity)

    ################################
    ##   Custom FDB Predicates    ##
    ################################

    @fdb.transactional
    def _get_custom_predicate(self, tr, prefix_tuple, partial_key):
        if partial_key:
            for k, _ in tr[self.subspace.range(prefix_tuple)]:
                yield k
        else:
            k = self.subspace.pack(prefix_tuple)
            if tr[k].present():
                yield k

    def _resolve_custom_fdb(self, predicate, arity):
        str_arity = str(arity)
        prefix = predicate+str_arity

        def func(*args):
            assert len(args) == arity, "arity mismatch"
            leftmost_consts = [arg.id for arg in
                               itertools.takewhile(lambda x: x.is_const(), args)]
            prefix_tuple = (prefix,) + tuple(leftmost_consts)
            partial_key = len(leftmost_consts) < arity
            for t in self._get_custom_predicate(db, prefix_tuple, partial_key):
                yield self.subspace.unpack(t)[1:]

        pyEngine.Python_resolvers[prefix+'/'+str_arity] = func

    def create_custom_fdb(self, predicates):
        # predicates should be a list of form [('pred_name', arity)]
        for predicate, arity in predicates:
            self._resolve_custom_fdb(predicate, arity)
