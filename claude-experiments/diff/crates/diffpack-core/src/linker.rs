//! Framework-neutral graph liveness and export-demand propagation.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::module_graph::DenseModuleId;
use crate::transform::{DependencyDemand, ModuleLiveness};
use crate::tree_shake::Demand;

pub struct LinkDependency<'a> {
    pub specifier: &'a str,
    pub target: DenseModuleId,
    pub demand: &'a DependencyDemand,
}

pub trait LinkGraph {
    type Dependencies<'a>: Iterator<Item = LinkDependency<'a>>
    where
        Self: 'a;

    fn module_count(&self) -> usize;
    fn entry(&self) -> DenseModuleId;
    fn present(&self, module: DenseModuleId) -> bool;
    fn droppable(&self, module: DenseModuleId) -> bool;
    fn liveness(&self, module: DenseModuleId) -> Option<&ModuleLiveness>;
    fn dependencies(&self, module: DenseModuleId) -> Self::Dependencies<'_>;
}

/// Determines which reachable modules must execute after export-level DCE.
pub fn live_modules(
    graph: &impl LinkGraph,
    reachable: &HashSet<DenseModuleId>,
) -> HashSet<DenseModuleId> {
    let mut live = vec![false; graph.module_count()];
    let mut used = vec![Demand::default(); graph.module_count()];
    let mut queue = VecDeque::new();

    fn mark_live(
        index: DenseModuleId,
        reachable: &HashSet<DenseModuleId>,
        live: &mut [bool],
        queue: &mut VecDeque<DenseModuleId>,
    ) {
        if reachable.contains(&index) && !live[index] {
            live[index] = true;
            queue.push_back(index);
        }
    }

    fn add_used(
        index: DenseModuleId,
        all: bool,
        names: &[String],
        reachable: &HashSet<DenseModuleId>,
        live: &mut [bool],
        used: &mut [Demand],
        queue: &mut VecDeque<DenseModuleId>,
    ) {
        if !reachable.contains(&index) {
            return;
        }
        let mut changed = false;
        if all && !used[index].all {
            used[index].all = true;
            changed = true;
        }
        for name in names {
            changed |= used[index].names.insert(name.clone());
        }
        if changed {
            live[index] = true;
            queue.push_back(index);
        }
    }

    let entry = graph.entry();
    if reachable.contains(&entry) {
        live[entry] = true;
        used[entry].all = true;
        queue.push_back(entry);
    }

    while let Some(source) = queue.pop_front() {
        if !graph.present(source) {
            continue;
        }
        let dependencies = graph.dependencies(source).collect::<Vec<_>>();
        let targets = dependencies
            .iter()
            .map(|dependency| (dependency.specifier, dependency.target))
            .collect::<HashMap<_, _>>();

        for dependency in &dependencies {
            if dependency.demand.dynamic {
                mark_live(dependency.target, reachable, &mut live, &mut queue);
                add_used(
                    dependency.target,
                    true,
                    &[],
                    reachable,
                    &mut live,
                    &mut used,
                    &mut queue,
                );
            }
        }
        for dependency in &dependencies {
            if !dependency.demand.deferred()
                && graph.present(dependency.target)
                && !graph.droppable(dependency.target)
            {
                mark_live(dependency.target, reachable, &mut live, &mut queue);
            }
        }

        let liveness = graph.liveness(source).expect("present link module");
        let empty_liveness = liveness.exports.is_empty()
            && liveness.reexports.is_empty()
            && liveness.star_reexports.is_empty()
            && liveness.body_uses.is_empty();
        if empty_liveness {
            for dependency in &dependencies {
                if !dependency.demand.deferred() {
                    add_used(
                        dependency.target,
                        dependency.demand.all,
                        &dependency.demand.names,
                        reachable,
                        &mut live,
                        &mut used,
                        &mut queue,
                    );
                }
            }
            continue;
        }

        for body_use in &liveness.body_uses {
            if let Some(&target) = targets.get(body_use.specifier.as_str()) {
                add_used(
                    target,
                    body_use.all,
                    &body_use.names,
                    reachable,
                    &mut live,
                    &mut used,
                    &mut queue,
                );
            }
        }
        let source_all = used[source].all;
        let source_names = used[source].names.clone();
        for reexport in &liveness.reexports {
            if (source_all || source_names.contains(&reexport.exported))
                && let Some(&target) = targets.get(reexport.specifier.as_str())
            {
                if reexport.imported == "*" {
                    add_used(
                        target,
                        true,
                        &[],
                        reachable,
                        &mut live,
                        &mut used,
                        &mut queue,
                    );
                } else {
                    add_used(
                        target,
                        false,
                        std::slice::from_ref(&reexport.imported),
                        reachable,
                        &mut live,
                        &mut used,
                        &mut queue,
                    );
                }
            }
        }
        for specifier in &liveness.star_reexports {
            let Some(&target) = targets.get(specifier.as_str()) else {
                continue;
            };
            if source_all {
                add_used(
                    target,
                    true,
                    &[],
                    reachable,
                    &mut live,
                    &mut used,
                    &mut queue,
                );
            } else {
                let names = source_names
                    .iter()
                    .filter(|name| name.as_str() != "default" && !liveness.exports.contains(name))
                    .cloned()
                    .collect::<Vec<_>>();
                if !names.is_empty() {
                    add_used(
                        target, false, &names, reachable, &mut live, &mut used, &mut queue,
                    );
                }
            }
        }
    }

    live.into_iter()
        .enumerate()
        .filter_map(|(module, live)| live.then_some(module))
        .collect()
}

pub fn export_demands(graph: &impl LinkGraph, sources: &[DenseModuleId]) -> Vec<Demand> {
    let mut demands = vec![Demand::default(); graph.module_count()];
    for &source in sources {
        if !graph.present(source) {
            continue;
        }
        for dependency in graph.dependencies(source) {
            demands[dependency.target].merge(Demand {
                all: dependency.demand.all || dependency.demand.dynamic,
                names: dependency.demand.names.iter().cloned().collect(),
            });
        }
    }
    demands
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Module {
        droppable: bool,
        liveness: ModuleLiveness,
        dependencies: Vec<(String, DenseModuleId, DependencyDemand)>,
    }

    struct Graph(Vec<Option<Module>>);

    fn dependency_view(
        dependency: &(String, DenseModuleId, DependencyDemand),
    ) -> LinkDependency<'_> {
        LinkDependency {
            specifier: &dependency.0,
            target: dependency.1,
            demand: &dependency.2,
        }
    }

    impl LinkGraph for Graph {
        type Dependencies<'a>
            = std::iter::Map<
            std::slice::Iter<'a, (String, DenseModuleId, DependencyDemand)>,
            fn(&'a (String, DenseModuleId, DependencyDemand)) -> LinkDependency<'a>,
        >
        where
            Self: 'a;

        fn module_count(&self) -> usize {
            self.0.len()
        }
        fn entry(&self) -> DenseModuleId {
            0
        }
        fn present(&self, module: DenseModuleId) -> bool {
            self.0.get(module).is_some_and(Option::is_some)
        }
        fn droppable(&self, module: DenseModuleId) -> bool {
            self.0[module].as_ref().unwrap().droppable
        }
        fn liveness(&self, module: DenseModuleId) -> Option<&ModuleLiveness> {
            self.0[module].as_ref().map(|module| &module.liveness)
        }
        fn dependencies(&self, module: DenseModuleId) -> Self::Dependencies<'_> {
            self.0[module]
                .as_ref()
                .unwrap()
                .dependencies
                .iter()
                .map(dependency_view)
        }
    }

    fn edge(
        specifier: &str,
        target: usize,
        all: bool,
        dynamic: bool,
    ) -> (String, usize, DependencyDemand) {
        (
            specifier.to_string(),
            target,
            DependencyDemand {
                specifier: specifier.to_string(),
                all,
                dynamic,
                import_syntax: true,
                ..DependencyDemand::default()
            },
        )
    }

    #[test]
    fn dynamic_edges_keep_the_namespace_while_unused_droppable_edges_die() {
        let graph = Graph(vec![
            Some(Module {
                dependencies: vec![
                    edge("./unused", 1, false, false),
                    edge("./lazy", 2, false, true),
                ],
                ..Module::default()
            }),
            Some(Module {
                droppable: true,
                ..Module::default()
            }),
            Some(Module {
                droppable: true,
                ..Module::default()
            }),
        ]);
        let reachable = HashSet::from([0, 1, 2]);
        assert_eq!(live_modules(&graph, &reachable), HashSet::from([0, 2]));
        let demands = export_demands(&graph, &[0]);
        assert!(!demands[1].all);
        assert!(demands[2].all);
    }
}
