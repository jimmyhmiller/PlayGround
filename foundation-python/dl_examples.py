import itertools
import random

import fdb
import fdb.tuple

from pyDatalog import pyDatalog
from datalog import Datalog, kvp

fdb.api_version(300)
db = fdb.open()

# app is a subspace for an open directory
app = fdb.directory.create_or_open(db, ('app,'))
Datalog(app, 10)

############################
##   Relational Algebra   ##
############################

pyDatalog.create_atoms('q,r,s,X,Y,Z')

# Select
r(X,'foo')

# Project
q(X) <= r(X,Y)

# Join
q(X, Y, Z) <= r(X, Y) & s(X, Z)

###################################
##   Beyond Relational Algebra   ##
###################################

pyDatalog.create_atoms('closure,edge,Source,Target,Intermediate')

@fdb.transactional
def set_cycle(tr, size, label=''):
    last = size-1
    for i in range(last):
        tr[app.pack(('edge', label+str(i), label+str(i+1)))] = ''
    tr[app.pack(('edge', label+str(last), label+'0'))] = ''

edge(Source, Target) <= kvp('edge', Source, Target, '')

# Transitive closure (left-recursive)

closure(Source, Target) <= edge(Source, Target)
closure(Source, Target) <= closure(Source, Intermediate) & edge(Intermediate, Target)

# Strongly connected components

pyDatalog.create_atoms('strongly_connected')

strongly_connected(Source, Target) <= closure(Source, Target) & closure(Target, Source)

@fdb.transactional
def set_bicycle(tr, size, label1, label2):
    del tr[app.range(('edge',))]
    set_cycle(tr, size, label1)
    set_cycle(tr, size, label2)
    tr[app.pack(('edge', label1+'0', label2+'0'))] = ''

# Bipartite graphs

pyDatalog.create_atoms('odd_path, bipartite, Mid1, Mid2')

odd_path(Source, Target) <= edge(Source, Target)
odd_path(Source, Target) <= odd_path(Source, Mid1) & \
                            edge(Mid1, Mid2) & edge(Mid2, Target)

bipartite() <= ~odd_path(Source, Source)


# Same generation in genealogy

pyDatalog.create_atoms('same_generation,child,parent,attends_school')
pyDatalog.create_atoms('Child,Child1,Child2,Parent,Parent1,Parent2,School')

same_generation(Child1, Child2) <= child(Child1, Parent) & parent(Parent, Child2)
same_generation(Child1, Child2) <= same_generation(Parent1, Parent2) & \
                                   child(Child1, Parent2) & parent(Parent2, Child2)

child(Child, Parent) <= kvp('child', Child, Parent, '')

parent(Parent, Child) <= kvp('parent', Parent, Child, '')

@fdb.transactional
def set_parent_child(tr, parent, child):
    tr[app.pack(('parent', parent, child))] = ''
    tr[app.pack(('child', child, parent))] = ''

@fdb.transactional
def set_genealogy(tr):
    del tr[app.range(('parent',))]
    del tr[app.range(('child',))]

    set_parent_child(tr, 'Henry', 'Mark')
    set_parent_child(tr, 'Henry', 'Karen')
    set_parent_child(tr, 'Mark', 'Joe')
    set_parent_child(tr, 'Karen', 'Susan')
    set_parent_child(tr, 'Joe', 'Frank')

attends_school(Child, School) <= kvp('attends_school', Child, School, '')

@fdb.transactional
def set_school(tr):
    tr[app.pack(('attends_school', 'Susan', 'Pine Elementary'))] = ''
    tr[app.pack(('attends_school', 'Frank', 'Pine Elementary'))] = ''

# Functions and lists (Daedalus puzzle)

pyDatalog.create_atoms('path,legal,legal_move')
pyDatalog.create_atoms('Start,End,Mid,Path,Path1,Move,A,B')

(path[Start, End] == Path) <= edge(Start, End, Move) & (Path == (Move,))
(path[Start, End] == Path) <= (path[Start, Mid] == Path1) & edge(Mid, End, Move) & \
                              (Path == Path1+(Move,))

edge(Start, End, Move) <= legal_move(Start, A, B) & (Move == (A, B)) & \
                (End == Start[0:A]+Start[B:B+1]+Start[A+1:B]+Start[A:A+1]+Start[B+1:])

legal_move(Start, A, B) <= kvp('action', A, B, '') & legal(Start, A, B)

legal(Start, A, B) <= (Start[A] == '*')
legal(Start, A, B) <= (Start[B] == '*')

@fdb.transactional
def set_operations(tr):
    del tr[app.range(('action',))]
    tr[app.pack(('action', 0, 1))] = ''
    tr[app.pack(('action', 0, 3))] = ''
    tr[app.pack(('action', 1, 2))] = ''
    tr[app.pack(('action', 1, 4))] = ''
    tr[app.pack(('action', 2, 5))] = ''
    tr[app.pack(('action', 3, 4))] = ''
    tr[app.pack(('action', 3, 6))] = ''
    tr[app.pack(('action', 4, 5))] = ''
    tr[app.pack(('action', 4, 7))] = ''
    tr[app.pack(('action', 5, 8))] = ''
    tr[app.pack(('action', 6, 7))] = ''
    tr[app.pack(('action', 7, 8))] = ''

# Example: solve_dedalus('HAAHS*T*G', 'HASHTAG**')
def solve_dedalus(start, end):
    print(path[list(start), list(end)] == Path)

# Aggregation (job matching)

pyDatalog.create_atoms('has_skill,lives_in,in_location,requires,match')
pyDatalog.create_atoms('matching_skill,num_matching_skills,num_reqs')
pyDatalog.create_atoms('best_jobs,best_candidates')
pyDatalog.create_atoms('Candidate,Job,Skill,City,Score')

has_skill(Candidate,Skill) <= kvp('has_skill',Candidate, Skill, '')
lives_in(Candidate,City) <= kvp('lives_in', Candidate, City, '')
in_location(Job,City) <= kvp('in_location', Job, City, '')
requires(Job, Skill) <= kvp('requires', Job, Skill, '')

matching_skill(Candidate, Job, Skill) <= has_skill(Candidate, Skill) & \
                                         requires(Job, Skill)

match(Candidate, Job) <= matching_skill(Candidate, Job, Skill) & \
                        lives_in(Candidate, City) & in_location(Job, City)

(num_matching_skills[Candidate, Job] == len_(Skill)) <= \
                                        matching_skill(Candidate, Job, Skill)

(num_reqs[Job] == len_(Skill)) <= requires(Job, Skill)

match(Candidate, Job, Score) <= match(Candidate,Job) & \
                          (Score == num_matching_skills[Candidate, Job]/num_reqs[Job])

(best_jobs[Candidate] == concat_(Job, order_by=Score, sep=',')) <= \
                                                          match(Candidate, Job, Score)

(best_candidates[Job] == concat_(Candidate, order_by=Score, sep=',')) <= \
                                                          match(Candidate, Job, Score)

@fdb.transactional
def clear_job_data(tr):
    del tr[app.range(('has_skill',))]
    del tr[app.range(('lives_in',))]
    del tr[app.range(('requires',))]
    del tr[app.range(('in_location',))]

@fdb.transactional
def set_job_record(tr, pred, entity, value):
    tr[app.pack((pred, entity, value))] = ''

def set_job_data():
    names = ['Joe', 'Henry', 'Susan', 'Mark', 'Joanna']
    companies = ['Zandian', 'Xanath', 'Trianon', 'Micromoves', 'Prionics']

    name_product = itertools.product(names, range(10))
    company_product = itertools.product(companies, range(10))
    candidates = [name+str(num) for name, num in name_product]
    jobs = [company+'_'+str(num) for company, num in company_product]

    skills = ['Java', 'Scala', 'Python', 'R', 'C++']
    locations = ['San Francisco', 'DC', 'New York City', 'Boston', 'Austin']

    clear_job_data(db)

    for candidate in candidates:
        cand_skills = random.sample(skills, random.randint(1, len(skills)))
        for skill in cand_skills:
            set_job_record(db, 'has_skill', candidate, skill)
        set_job_record(db, 'lives_in', candidate, random.choice(locations))

    for job in jobs:
        job_skills = random.sample(skills, random.randint(1, len(skills)))
        for skill in job_skills:
            set_job_record(db, 'requires', job, skill)
        set_job_record(db, 'in_location', job, random.choice(locations))