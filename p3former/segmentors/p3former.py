import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from datetime import datetime
import torch
from mmdet3d.registry import MODELS
from mmdet3d.models.segmentors.cylinder3d import Cylinder3D
from mmdet3d.structures import PointData
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig

@MODELS.register_module()
class _P3Former(Cylinder3D):
    """P3Former."""

    def __init__(self,
                 voxel_encoder: ConfigType,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 offset_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 loss_regularization: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_offset: bool = True) -> None:
        super().__init__(voxel_encoder=voxel_encoder,
                        backbone=backbone,
                        decode_head=decode_head,
                        neck=neck,
                        auxiliary_head=auxiliary_head,
                        loss_regularization=loss_regularization,
                        train_cfg=train_cfg,
                        test_cfg=test_cfg,
                        data_preprocessor=data_preprocessor,
                        init_cfg=init_cfg)
        self.use_offset = use_offset
        if self.use_offset:
            self.offset_head = MODELS.build(offset_head)
    def loss(self, batch_inputs_dict,batch_data_samples):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        # extract features using backbone
        x, x_ins = self.extract_feat(batch_inputs_dict)
        
        batch_inputs_dict['features'] = x.features
        losses = dict()
        loss_decode = self._decode_head_forward_train(batch_inputs_dict, batch_data_samples)
        losses.update(loss_decode)
        if self.use_offset:
            pred_offsets = self.offset_head(x_ins, batch_inputs_dict)
            offset_loss = self.offset_loss(pred_offsets, batch_data_samples)
            losses['offset_loss'] = sum(offset_loss)
        validate_offset(pred_offsets, batch_inputs_dict, batch_data_samples, vis=True)
        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        x, x_ins = self.extract_feat(batch_inputs_dict)
        batch_inputs_dict['features'] = x.features
        pred_offsets = self.offset_head(x_ins, batch_inputs_dict)
        validate_offset(pred_offsets, batch_inputs_dict, batch_data_samples, vis=True)
        pts_semantic_preds, pts_instance_preds = self.decode_head.predict(batch_inputs_dict, batch_data_samples)
        return self.postprocess_result(pts_semantic_preds, pts_instance_preds, batch_data_samples)

    def postprocess_result(self, pts_semantic_preds, pts_instance_preds, batch_data_samples):
        for i in range(len(pts_semantic_preds)):
            batch_data_samples[i].set_data(
                {'pred_pts_seg': PointData(**{'pts_semantic_mask': pts_semantic_preds[i],
                                                'pts_instance_mask': pts_instance_preds[i]})})
        return batch_data_samples
    
    def offset_loss(self, pred_offsets, batch_data_samples):
        loss_list_list = []
        for i, b in enumerate(batch_data_samples):
            valid = torch.from_numpy(batch_data_samples[0].gt_pts_seg.pts_valid).cuda()
            gt_offsets = b.gt_pts_seg.pts_offsets
            pt_diff = pred_offsets[i] - gt_offsets   # (N, 3)
            pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
            valid = valid.view(-1).float()
            offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)
            loss_len = len((loss_list_list,))
            if len(loss_list_list) < loss_len:
                loss_list_list = [[] for j in range(loss_len)]
            for j in range(loss_len):
                loss_list_list[j].append((offset_norm_loss,)[j])
        mean_loss_list = []
        for i in range(len(loss_list_list)):
            mean_loss_list.append(torch.mean(torch.stack(loss_list_list[i])))
        return mean_loss_list

def validate_offset(pred_offsets, batch_inputs_dict, batch_data_samples, vis=True):
    valid = batch_data_samples[0].gt_pts_seg.pts_valid
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    pts = batch_inputs_dict['points'][0].cpu().numpy()
    draw_point(pts, name=f'pcl_gt_img_{time}.png')
    draw_point(pts[valid], name=f'pcl_thing_img_{time}.png')

    embedding = [offset + xyz for offset, xyz in zip(pred_offsets, batch_inputs_dict['points'][0][:, :3])]
    for emb in embedding:
        with torch.no_grad():
            shifted_points = emb.cpu().numpy()
            draw_point(shifted_points, name=f'shifted_pointcloud_{time}.png')
            draw_point(shifted_points[valid], name=f'shifted_thing_pointcloud_{time}.png')


def draw_point(pcl, indices=None, name='pcl_img.png', s=1):
    # Assuming `embedding` is your point cloud and `indices` are the indices of the centroids
    embedding_np = pcl  # Convert to numpy array for plotting
    
    # Create a new color array
    colors = np.full(embedding_np.shape[0], 'w')  # All points are blue
    
    # Create a new size array
    sizes = np.full(embedding_np.shape[0], s)  # All points are size 1
    
    if indices is not None:
        indices_np = indices.flatten()  # Flatten the indices array for indexing
        colors[indices_np] = 'r'  # Centroid points are red
        sizes[indices_np] = 10  # Centroid points are size 20

    # Create a 3D plot
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot()

    # Plot the point cloud in 3D
    scatter = ax.scatter(embedding_np[:, 0], embedding_np[:, 1], c=colors, s=sizes)

    # Set background color to black
    ax.set_facecolor('black')
    ax.set_aspect('auto')
    # ax.set_axis_off()

    # Hide grid
    # ax.grid(False)

    # Save the plot to an image file
    plt.savefig(name)

def write_img(img, name='heatmap.png'):
    plt.imsave(name, img)

def find_peaks(heatmap, threshold):
    # Use maximum filter to find local maxima
    local_max = maximum_filter(heatmap, footprint=np.ones((3, 3)), mode='constant') == heatmap
    write_img(local_max, name='localmax.png')
    # Apply threshold to the local maxima
    peaks = local_max & (heatmap > threshold)
    return np.argwhere(peaks)

def draw_point_with(pcl, indices=None, name='pcl_img.png', s=1, pcl2=None):
    # Assuming `embedding` is your point cloud and `indices` are the indices of the centroids
    embedding_np = pcl  # Convert to numpy array for plotting
    
    # Create a new color array
    colors = np.full(embedding_np.shape[0], 'w')  # All points are blue
    
    # Create a new size array
    sizes = np.full(embedding_np.shape[0], s)  # All points are size 1
    
    if indices is not None:
        indices_np = indices.flatten()  # Flatten the indices array for indexing
        colors[indices_np] = 'r'  # Centroid points are red
        sizes[indices_np] = 10  # Centroid points are size 20

    # Create a 3D plot
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot()

    

    # Plot the point cloud in 3D
    scatter = ax.scatter(embedding_np[:, 0], embedding_np[:, 1], c=colors, s=sizes)
    if pcl2 is not None:
        scatter = ax.scatter(pcl2[:, 0], pcl2[:, 1], c='r', s=s)

    # Set background color to black
    ax.set_facecolor('black')
    ax.set_aspect('auto')
    # ax.set_axis_off()

    # Hide grid
    # ax.grid(False)

    # Save the plot to an image file
    plt.savefig(name)