import torch
import torch.utils.data
from torch_geometric.data import Data


class Batch(Data):
    r""" merge many of small graph to a big graph
    """

    def __init__(self, **kwargs):
        super(Batch, self).__init__(**kwargs)

    @staticmethod
    def from_data_list(data_list):

        # data_list: [[],[],[]]
        # []: (pos, neg, neg, ..., neg) Note: one pos, many neg

        r""" re-assign edge index, due to the edge index must be unique in one graph.
        Concretely, add offset to each edge index of the small graph, and to construct a big graph.
        """

        keys = data_list[0][0].keys
        assert 'slices_indicator' not in keys

        # slices_indicator is number of nodes in each graph
        batch = Batch()
        batch.__data_class__ = data_list[0].__class__
        for key in keys + ['slices_indicator']:
            batch[key] = []

        # keep list structure, only use in processing of the raw dataset
        # print(batch)
        delattr(batch, "file_items")
        if hasattr(batch, "outfit_cates"):
            # print("outfit_cates",batch['outfit_cates'])
            delattr(batch, "outfit_cates")
            # print(batch)


        # record the offset of the edge index
        shift = 0
        for pair_data in data_list:
            for data in pair_data:
                for key in batch.keys:
                    if key == 'slices_indicator':
                        batch[key].append(torch.tensor([shift], dtype=torch.int))
                    elif key == 'edge_index':
                        batch[key].append(data[key] + shift) #
                    else:
                        batch[key].append(data[key])
                shift += data.num_nodes
        batch['slices_indicator'].append(torch.tensor([shift], dtype=torch.int))

        tmp_data = data_list[0][0]
        for key in batch.keys:
            if key=='rcid_index' or key=='c':
                continue
            cat_dim = tmp_data.__cat_dim__(key, tmp_data[key])
            # torch.from_numpy(batch[key])   #这里修改过
            batch[key] = torch.cat(batch[key], dim=cat_dim)

        batch['edge_weight'].unsqueeze_(-1)

        return batch.contiguous()  #改变内存存储顺序，方便view的使用

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        if hasattr(self, 'slices_indicator'):
            return len(self.__getitem__('slices_indicator')) - 1
        return None


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=Batch.from_data_list, **kwargs)
