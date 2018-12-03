

// https://github.com/frankmcsherry/blog/blob/master/posts/2018-05-19.md

use std::rc::Rc;
use std::cell::RefCell;


/// A sorted list of distinct tuples.
#[derive(Debug)]
pub struct Relation<Tuple: Ord> {
    elements: Vec<Tuple>
}


impl<Tuple: Ord, I: IntoIterator<Item=Tuple>> From<I> for Relation<Tuple> {
    fn from(iterator: I) -> Self {
        let mut elements: Vec<Tuple> = iterator.into_iter().collect();
        elements.sort_unstable();
        elements.dedup();
        Relation { elements }
    }
}

impl<Tuple: Ord> Relation<Tuple> {


    fn from_vec(mut elements: Vec<Tuple>) -> Self {
        elements.sort_unstable();
        elements.dedup();
        Relation { elements }
    }

    /// Merges two relations into their union.
    pub fn merge(self, other: Self) -> Self {
        let mut elements = Vec::with_capacity(self.elements.len() + other.elements.len());
        elements.extend(self.elements.into_iter());
        elements.extend(other.elements.into_iter());
        elements.into()
    }
}

impl<Tuple: Ord> Clone for Variable<Tuple> {
    fn clone(&self) -> Self {
        Variable {
            distinct: self.distinct,
            name: self.name.clone(),
            stable: self.stable.clone(),
            recent: self.recent.clone(),
            to_add: self.to_add.clone(),
        }
    }
}

pub struct Variable<Tuple: Ord> {

        /// Should the variable be maintained distinctly.
    distinct: bool,
    /// A useful name for the variable.
    name: String,
    /// A list of already processed tuples.
    stable: Rc<RefCell<Vec<Relation<Tuple>>>>,
    /// A list of recently added but unprocessed tuples.
    recent: Rc<RefCell<Relation<Tuple>>>,
    /// A list of tuples yet to be introduced.
    to_add: Rc<RefCell<Vec<Relation<Tuple>>>>,
}

impl<Tuple: Ord> std::ops::Deref for Relation<Tuple> {
    type Target = [Tuple];
    fn deref(&self) -> &Self::Target {
        &self.elements[..]
    }
}

impl<Tuple: Ord> Variable<Tuple> {

    fn complete(self) -> Relation<Tuple> {

        assert!(self.recent.borrow().is_empty());
        assert!(self.to_add.borrow().is_empty());
        let mut result: Relation<Tuple> = Vec::new().into();
        while let Some(batch) = self.stable.borrow_mut().pop() {
            result = result.merge(batch);
        }
        result
    }

    fn from_join<K: Ord, V1: Ord, V2: Ord>(
            &self,
            input1: &Variable<(K,V1)>,
            input2: &Variable<(K,V2)>,
            logic: impl FnMut(&K,&V1,&V2)->Tuple) {
            
        join_into(input1, input2, self, logic)
    }

    fn from_map<T2: Ord>(&self, input: &Variable<T2>, logic: impl FnMut(&T2)->Tuple) {
        map_into(input, self, logic)
    }

    fn new(name: &str) -> Self {
        Variable {
            distinct: true,
            name: name.to_string(),
            stable: Rc::new(RefCell::new(Vec::new().into())),
            recent: Rc::new(RefCell::new(Vec::new().into())),
            to_add: Rc::new(RefCell::new(Vec::new().into())),
        }
    }


    /// Inserts a relation into the variable.
    ///
    /// This is most commonly used to load initial values into a variable.
    /// it is not obvious that it should be commonly used otherwise, but
    /// it should not be harmful.
    pub fn insert(&self, relation: Relation<Tuple>) {
        if !relation.is_empty() {
            self.to_add.borrow_mut().push(relation);
        }
    }
}


fn join_helper<Key: Ord, Val1: Ord, Val2: Ord>(
    input1: &Relation<(Key,Val1)>,
    input2: &Relation<(Key,Val2)>,
    mut result: impl FnMut(&Key, &Val1, &Val2)) {
    let mut slice1 = &input1.elements[..];
    let mut slice2 = &input2.elements[..];

    while !slice1.is_empty() && !slice2.is_empty() {

        use std::cmp::Ordering;

        // If the keys match call `result`, else advance the smaller key until they might.
        match slice1[0].0.cmp(&slice2[0].0) {
            Ordering::Less => {
                slice1 = gallop(slice1, |x| x.0 < slice2[0].0);
            },
            Ordering::Equal => {

                // Determine the number of matching keys in each slice.
                let count1 = slice1.iter().take_while(|x| x.0 == slice1[0].0).count();
                let count2 = slice2.iter().take_while(|x| x.0 == slice2[0].0).count();

                // Produce results from the cross-product of matches.
                for index1 in 0 .. count1 {
                    for index2 in 0 .. count2 {
                        result(&slice1[0].0, &slice1[index1].1, &slice2[index2].1);
                    }
                }

                // Advance slices past this key.
                slice1 = &slice1[count1..];
                slice2 = &slice2[count2..];
            }
            Ordering::Greater => {
                slice2 = gallop(slice2, |x| x.0 < slice1[0].0);
            }
        }
    }

}

pub fn gallop<T>(mut slice: &[T], mut cmp: impl FnMut(&T)->bool) -> &[T] {
    // if empty slice, or already >= element, return
    if slice.len() > 0 && cmp(&slice[0]) {
        let mut step = 1;
        while step < slice.len() && cmp(&slice[step]) {
            slice = &slice[step..];
            step = step << 1;
        }

        step = step >> 1;
        while step > 0 {
            if step < slice.len() && cmp(&slice[step]) {
                slice = &slice[step..];
            }
            step = step >> 1;
        }

        slice = &slice[1..]; // advance one, as we always stayed < value
    }

    return slice;
}

pub fn join_into<Key: Ord, Val1: Ord, Val2: Ord, Result: Ord>(
    input1: &Variable<(Key, Val1)>,
    input2: &Variable<(Key, Val2)>,
    output: &Variable<Result>,
    mut logic: impl FnMut(&Key, &Val1, &Val2)->Result) {

    let mut results = Vec::new();

    // input1.recent and input2.stable.
    for batch2 in input2.stable.borrow().iter() {
        join_helper(&input1.recent.borrow(), &batch2, |k,v1,v2| results.push(logic(k,v1,v2)));
    }

    // input1.stable and input2.recent.
    for batch1 in input1.stable.borrow().iter() {
        join_helper(&batch1, &input2.recent.borrow(), |k,v1,v2| results.push(logic(k,v1,v2)));
    }

    // input1.recent and input2.recent.
    join_helper(&input1.recent.borrow(), &input2.recent.borrow(), |k,v1,v2| results.push(logic(k,v1,v2)));

    output.insert(results.into());
}

pub fn map_into<T1: Ord, T2: Ord>(
    input: &Variable<T1>,
    output: &Variable<T2>,
    mut logic: impl FnMut(&T1)->T2) {

    let mut results = Vec::new();
    let recent = input.recent.borrow();
    for tuple in recent.iter() {
        results.push(logic(tuple));
    }

    output.insert(Relation::from_vec(results));
}


impl<Tuple: Ord> VariableTrait for Variable<Tuple> {

    fn changed(&mut self) -> bool {

        // 1. Merge self.recent into self.stable.
        if !self.recent.borrow().is_empty() {
            let mut recent = ::std::mem::replace(&mut (*self.recent.borrow_mut()), Vec::new().into());
            while self.stable.borrow().last().map(|x| x.len() <= 2 * recent.len()) == Some(true) {
                let last = self.stable.borrow_mut().pop().unwrap();
                recent = recent.merge(last);
            }
            self.stable.borrow_mut().push(recent);
        }

        // 2. Move self.to_add into self.recent.
        let to_add = self.to_add.borrow_mut().pop();
        if let Some(mut to_add) = to_add {
            while let Some(to_add_more) = self.to_add.borrow_mut().pop() {
                to_add = to_add.merge(to_add_more);
            }
            // 2b. Restrict `to_add` to tuples not in `self.stable`.
            if self.distinct {
                for batch in self.stable.borrow().iter() {
                    let mut slice = &batch[..];
                    // Only gallop if the slice is relatively large.
                    if slice.len() > 4 * to_add.elements.len() {
                        to_add.elements.retain(|x| {
                            slice = gallop(slice, |y| y < x);
                            slice.len() == 0 || &slice[0] != x
                        });
                    }
                    else {
                        to_add.elements.retain(|x| {
                            while slice.len() > 0 && &slice[0] < x {
                                slice = &slice[1..];
                            }
                            slice.len() == 0 || &slice[0] != x
                        });
                    }
                }
            }
            *self.recent.borrow_mut() = to_add;
        }
        !self.recent.borrow().is_empty()
    }
}


pub struct Iteration {
    variables: Vec<Box<VariableTrait>>,
}

impl Iteration {
    /// Create a new iterative context.
    pub fn new() -> Self {
        Iteration { variables: Vec::new() }
    }
    /// Reports whether any of the monitored variables have changed since
    /// the most recent call.
    pub fn changed(&mut self) -> bool {
        let mut result = false;
        for variable in self.variables.iter_mut() {
            if variable.changed() { 
                result = true;
                break;
            }
        }
        result
    }
    /// Creates a new named variable associated with the iterative context.
    pub fn variable<Tuple: Ord+'static>(&mut self, name: &str) -> Variable<Tuple> {
        let variable = Variable::new(name);
        self.variables.push(Box::new(variable.clone()));
        variable
    }
    /// Creates a new named variable associated with the iterative context.
    ///
    /// This variable will not be maintained distinctly, and may advertise tuples as
    /// recent multiple times (perhaps unboundedly many times).
    pub fn variable_indistinct<Tuple: Ord+'static>(&mut self, name: &str) -> Variable<Tuple> {
        let mut variable = Variable::new(name);
        variable.distinct = false;
        self.variables.push(Box::new(variable.clone()));
        variable
    }
}


trait VariableTrait {
    /// Reports whether the variable has changed since it was last asked.
    fn changed(&mut self) -> bool;
}


fn main() {
    let mut iteration1 = Iteration::new();

    let parent_of = iteration1.variable::<(&str, &str)>("parent_of");
    let child_of = iteration1.variable::<(&str, &str)>("child_of");
    let ancestor_of = iteration1.variable::<(&str, &str)>("ancestor_of");

    parent_of.insert((vec![("Archie", "Janice"), 
                           ("Janice", "Lemon"),
                           ("Janice", "Mango"),
                           ("Lemon", "Warms")]).into());

    while iteration1.changed() {
            
        child_of.from_map(&parent_of, |&(parent, child)| (child, parent));
        ancestor_of.from_join(&parent_of, &parent_of, |&a, &b, _c| (a,b));
        ancestor_of.from_join(&ancestor_of, &child_of, |_b, &a, &c| (c,a));

    }

    let ancestors = ancestor_of.complete();


     println!("Variable\t{:?}", ancestors);
}
