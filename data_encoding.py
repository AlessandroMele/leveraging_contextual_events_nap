from pandas import read_csv, to_datetime
import networkx as nx
from ast import literal_eval
from os.path import join
from os import makedirs
import json
from torch_geometric.data import Data
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import gc
from collections import defaultdict
from config import MAX_WORKERS, INPUT_PATH
from functools import partial
_shared_data = None


def _encode_var_1(case_ig):
    global _shared
    name, igs, ohe_activity, path = (
        _shared['name'],
        _shared['igs'],
        _shared['ohe_activity'],
        _shared['path'],
    )

    _, ig = case_ig

    prefix = Data()
    i, all_activities = 0, [d for n, d in ig.nodes(data=True)]
    while i < len(all_activities) - 2:
        prefix.case_id = ig.case_id
        prefix.set = ig.set
        prefix.y = torch.tensor([ohe_activity[all_activities[i + 1].get('activity')]], dtype=torch.float)
        prefix.next_activity = all_activities[i + 1].get('activity')
        prefix.prefix_size = i + 1
        prefix.concurrent_events = all_activities[i].get('n_concurrent_events')

        act_ids, activities = map(list, zip(*((node, data) for node, data in ig.nodes(data=True)
                                              if node <= i)))
        edges = list(ig.subgraph(act_ids).edges(data=False))

        inter_cases = literal_eval(all_activities[i].get('concurrent_events'))
        if inter_cases:
            for conc_case, conc_act_id in inter_cases:
                context_ig = next((g for case, g in igs if case == conc_case), None)
                context_activity = context_ig.nodes.get(conc_act_id)

                activities.append(context_activity)
                edges.append((act_ids[0], len(activities)-1))

        activities_x = list(map(lambda x:
                                ohe_activity[x.get('activity')] + literal_eval(x.get('temporal_features')),
                                activities))

        prefix.x = torch.tensor(activities_x, dtype=torch.float)
        prefix.edge_index = torch.tensor(edges, dtype=torch.long).T

        if i > 0:
            torch.save(prefix, join(path, f'{name}_{prefix.case_id}_{prefix.prefix_size}.pt'))

        i += 1


def _encode_var_2(case_ig):
    global _shared
    name, igs, ohe_activity, path = (
        _shared['name'],
        _shared['igs'],
        _shared['ohe_activity'],
        _shared['path'],
    )

    _, ig = case_ig

    prefix = Data()
    i, all_activities = 0, [d for n, d in ig.nodes(data=True)]
    while i < len(all_activities) - 2:
        prefix.case_id = ig.case_id
        prefix.set = ig.set
        prefix.y = torch.tensor([ohe_activity[all_activities[i + 1].get('activity')]], dtype=torch.float)
        prefix.next_activity = all_activities[i + 1].get('activity')
        prefix.prefix_size = i + 1
        prefix.concurrent_events = all_activities[i].get('n_concurrent_events')

        act_ids, activities = map(list, zip(*((node, data) for node, data in ig.nodes(data=True)
                                              if node <= i)))
        edges = list(ig.subgraph(act_ids).edges(data=False))

        inter_cases = literal_eval(all_activities[i].get('concurrent_events'))
        if inter_cases:
            # estraggo il nodo o i nodi che hanno un arco verso l'ultimo nodo del prefisso
            incoming_nodes = list(ig.predecessors(i))

            for conc_case, conc_act_id in inter_cases:
                context_ig = next((g for case, g in igs if case == conc_case), None)
                context_activity = context_ig.nodes.get(conc_act_id)

                activities.append(context_activity)

                for node in incoming_nodes:
                    edges.append((node, len(activities)-1))

        activities_x = list(map(lambda x:
                                ohe_activity[x.get('activity')] + literal_eval(x.get('temporal_features')),
                                activities))

        prefix.x = torch.tensor(activities_x, dtype=torch.float)
        prefix.edge_index = torch.tensor(edges, dtype=torch.long).T

        if i > 0:
            torch.save(prefix, join(path, f'{name}_{prefix.case_id}_{prefix.prefix_size}.pt'))

        i += 1


def _encode_var_3(case_ig):
    global _shared
    name, igs, ohe_activity, path = (
        _shared['name'],
        _shared['igs'],
        _shared['ohe_activity'],
        _shared['path'],
    )

    _, ig = case_ig

    prefix = Data()
    i, all_activities = 0, [d for n, d in ig.nodes(data=True)]
    while i < len(all_activities) - 2:
        item = {
            'case_id': ig.case_id,
            'set': ig.set,
            'y': torch.tensor([ohe_activity[all_activities[i+1].get('activity')]], dtype=torch.float),
            'next_activity': all_activities[i+1].get('activity'),
            'prefix_size': i+1
             }

        act_ids, activities = map(list, zip(*((node, data) for node, data in ig.nodes(data=True) if node <= i)))
        edges = list(ig.subgraph(act_ids).edges(data=False))

        activities_x = list(map(lambda x:
                                ohe_activity[x.get('activity')] + literal_eval(x.get('temporal_features')),
                                activities))

        prefix.x = torch.tensor(activities_x, dtype=torch.float)
        prefix.edge_index = torch.tensor(edges, dtype=torch.long).T

        # context events
        context = Data()
        inter_cases = literal_eval(all_activities[i].get('concurrent_events'))
        if inter_cases:
            context_activities = [activities[0]]
            context_edges = []
            for conc_case, conc_act_id in inter_cases:
                context_ig = next((g for case, g in igs if case == conc_case), None)

                context_activity = context_ig.nodes.get(conc_act_id)
                context_activities.append(context_activity)

                context_edges.append((0, len(context_activities)-1))

        else:
            context_activities = ['dummy']
            context_edges = [[0, 0]]

        context_x = list(map(lambda x: [0] * (len(ohe_activity)+3) if x == 'dummy'
        else ohe_activity[x.get('activity')] + literal_eval(x.get('temporal_features')), context_activities))
        context.x = torch.tensor(context_x, dtype=torch.float)
        context.edge_index = torch.tensor(context_edges, dtype=torch.long).T

        item.update({
            'concurrent_events': len(inter_cases),
            'prefix': prefix,
            'context': context
        })

        if i > 0:
            torch.save(item, join(path, f"{name}_{item.get('case_id')}_{item.get('prefix_size')}.pt"))

        i += 1


def _encode_var_4(case_ig):
    global _shared
    name, igs, ohe_activity, path = (
        _shared['name'],
        _shared['igs'],
        _shared['ohe_activity'],
        _shared['path'],
    )

    _, ig = case_ig

    prefix = Data()
    i, all_activities = 0, [d for n, d in ig.nodes(data=True)]
    while i < len(all_activities) - 2:
        item = {
            'case_id': ig.case_id,
            'set': ig.set,
            'y': torch.tensor([ohe_activity[all_activities[i+1].get('activity')]], dtype=torch.float),
            'next_activity': all_activities[i+1].get('activity'),
            'prefix_size': i+1
             }

        act_ids, activities = map(list, zip(*((node, data) for node, data in ig.nodes(data=True) if node <= i)))
        edges = list(ig.subgraph(act_ids).edges(data=False))

        activities_x = list(map(lambda x:
                                ohe_activity[x.get('activity')] + literal_eval(x.get('temporal_features')),
                                activities))

        prefix.x = torch.tensor(activities_x, dtype=torch.float)
        prefix.edge_index = torch.tensor(edges, dtype=torch.long).T

        # context events
        context = Data()
        inter_cases = literal_eval(all_activities[i].get('concurrent_events'))
        if inter_cases:
            # tengo solo la tupla il cui id del nodo è più grande (altrimenti replico ig)
            d = defaultdict(lambda: float("-inf"))
            for k, v in inter_cases:
                d[k] = max(d[k], v)

            inter_cases = list(d.items())

            context_activities = [activities[0]]
            context_edges = []
            for conc_case, conc_act_id in inter_cases:
                offset = len(context_activities)
                context_ig = next((g for case, g in igs if case == conc_case), None)
                context_ig = nx.relabel_nodes(context_ig, {n: n + offset for n in context_ig.nodes()})

                g_act_ids, g_activities = map(list, zip(*((node, data) for node, data in context_ig.nodes(data=True)
                                                                      if node <= int(conc_act_id)+offset)))
                context_activities.extend(g_activities)
                context_edges.extend(list(context_ig.subgraph(g_act_ids).edges(data=False)))

            start_context_nodes = [i for i, data in enumerate(context_activities) if data.get("activity").startswith("artificial_start")]
            # aggiungo un arco dal primo evento del prefisso corrente al primo evento di ogni prefisso del contesto
            _ = [context_edges.append((0, start)) for start in start_context_nodes[1:]]

        else:
            context_activities = ['dummy']
            context_edges = [[0, 0]]

        context_x = list(map(lambda x: [0] * (len(ohe_activity)+3) if x == 'dummy'
        else ohe_activity[x.get('activity')] + literal_eval(x.get('temporal_features')), context_activities))
        context.x = torch.tensor(context_x, dtype=torch.float)
        context.edge_index = torch.tensor(context_edges, dtype=torch.long).T

        item.update({
            'concurrent_events': len(inter_cases),
            'prefix': prefix,
            'context': context
        })

        if i > 0:
            torch.save(item, join(path, f"{name}_{item.get('case_id')}_{item.get('prefix_size')}.pt"))

        i += 1


def encode_prefix_igs(igs, shared_items):
    if shared_items.get('variant') == 'var_1':
        f = partial(_encode_var_1)
    elif shared_items.get('variant') == 'var_2':
        f = partial(_encode_var_2)
    elif shared_items.get('variant') == 'var_3':
        f = partial(_encode_var_3)
    elif shared_items.get('variant') == 'var_4':
        f = partial(_encode_var_4)

    with ProcessPoolExecutor(max_workers=int(MAX_WORKERS/2), initializer=init_worker, initargs=(shared_items,)) as executor:
        for _ in tqdm(executor.map(f, igs), total=len(igs), desc='Encoding prefix-IGs'):
            pass

    with open(join(shared_items.get('path'), 'done'), 'w') as _:
        pass


def _discovery(case_id):
    global _shared
    cases_time, nodes, igs, activities_to_filter = (
        _shared['cases_time'],
        _shared['nodes'],
        _shared['igs'],
        _shared['activities_to_filter']
    )

    ig = igs.loc[igs['case_id'] == case_id]
    g = nx.MultiDiGraph()
    g.case_id, g.set = case_id, ig['set'].drop_duplicates().tolist()[0]

    for event in ig.itertuples(index=False):
        if event.type == 'v':
            start_time = to_datetime(event.start_time)
            active_cases = set(cases_time.loc[
                                   (event.set == cases_time['set']) &
                                   (start_time <= cases_time['max']) &
                                   (case_id != cases_time['case_id']), 'case_id'].tolist())

            concurrent_events = nodes.loc[(
                    (nodes['case_id'].isin(active_cases)) &
                    (~nodes['activity'].isin(activities_to_filter)) &
                    (nodes['start_time'] <= start_time) & (start_time <= nodes['end_time'])
            )]

            concurrent_nodes = list(zip(concurrent_events['case_id'], concurrent_events['node1']))
            n_concurrent_events = len(concurrent_events)

            temporal_features = [event.norm_time, event.trace_time, event.prev_event_time]

            g.add_node(event.node1,
                       activity=event.activity,
                       n_concurrent_events=n_concurrent_events,
                       concurrent_events=f'{concurrent_nodes}',
                       temporal_features=f'{temporal_features}'
                       )

        elif event.type == 'e':
            g.add_edge(event.node1, event.node2)

    return case_id, g


def discovery_inter_cases(cases, shared_items):
    f = partial(_discovery)
    igs = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker, initargs=(shared_items,)) as executor:
        for out in tqdm(executor.map(f, cases), total=len(cases), desc='Discovering inter cases'):
            if out is not None:
                igs.append(out)

    return igs


def get_ohe_encoding(name, items, attribute):
    try:
        with open(join(INPUT_PATH, f'{name}_ohe_{attribute}.json'), 'r') as f:
            ohe = json.load(f)

    except (Exception, ):
        ohe = {item:[1 if i == j else 0 for j in range(len(items))] for i, item in enumerate(items)}
        with open(join(INPUT_PATH, f'{name}_ohe_{attribute}.json'), 'w') as f:
            f.write(json.dumps(ohe, indent=1))

    return ohe


def init_worker(shared_data):
    global _shared
    _shared = shared_data


def main(name, variant):
    print('Reading dataset')
    igs = read_csv(join(INPUT_PATH, f'{name}_processed.g'), header=0, sep=',')

    nodes = igs.loc[igs['type'] == 'v']
    case_ids = nodes['case_id'].drop_duplicates().tolist()

    nodes['start_time'] = to_datetime(nodes['start_time'], format='ISO8601', utc=True)
    nodes['end_time'] = to_datetime(nodes['end_time'], format='ISO8601', utc=True)
    cases_time = nodes.groupby('case_id').agg(
        min=('end_time', 'min'),
        max=('end_time', 'max'),
        set=('set', 'first')
    ).reset_index()

    ohe_activity = get_ohe_encoding(name, nodes['activity'].drop_duplicates().tolist(), 'activity')
    activities_to_filter = ['artificial_start', 'artificial_end']

    shared_items = {
        'name': name,
        'igs': igs,
        'activities_to_filter': activities_to_filter,
        'cases_time': cases_time,
        'nodes': nodes,
        'ohe_activity': ohe_activity,
    }
    igs = discovery_inter_cases(case_ids, shared_items)

    del nodes, shared_items
    gc.collect()

    path = join('dataset', f'{name}_{variant}_tensors')
    makedirs(path, exist_ok=True)

    shared_items = {
        'variant': variant,
        'name': name,
        'igs': igs,
        'ohe_activity': ohe_activity,
        'path': path
    }

    encode_prefix_igs(igs, shared_items)

    del shared_items, igs
    gc.collect()
    