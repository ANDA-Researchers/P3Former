'''testing only'''
import matplotlib.pyplot as plt
from datetime import datetime
'''testing only'''

import numpy as np
import torch
import torch.nn.functional as F
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

        # Validate offset
        # validate_offset(pred_offsets, batch_inputs_dict, batch_data_samples, vis=True)
        for batch_i, p in enumerate(batch_inputs_dict['points']):
            batch_inputs_dict['points'][batch_i] = p[:,:3]
        assert len(pred_offsets[0]) == len(batch_inputs_dict['points'][0])

        # pred_offsets = [batch_data_samples[0].gt_pts_seg.pts_offsets]
        # Point shifting
        embedding = [offset + xyz for offset, xyz in zip(pred_offsets, batch_inputs_dict['points'])]

        # Get heatmap
        heatmap = []
        x_edges_list = []
        y_edges_list = []
        for i, emb in enumerate(embedding):
            valid = batch_data_samples[i].gt_pts_seg.pts_valid
            shifted_points = emb.detach()
            shifted_points = shifted_points.cpu().numpy()[valid]

            # Generate 2D heatmap
            min_val = np.min(emb.detach().cpu().numpy(), axis=0)
            max_val = np.max(emb.detach().cpu().numpy(), axis=0)
            grid_size = 0.1
            num_bins = np.ceil((max_val - min_val) / grid_size).astype(int)[:2]
            H, x_edges, y_edges = np.histogram2d(shifted_points[:, 0], shifted_points[:, 1], bins=num_bins)

            heatmap.append(H)
            x_edges_list.append(x_edges)
            y_edges_list.append(y_edges)

        window_size = 5
        threshold = 1
        heatmap_pooled = F.max_pool2d(torch.tensor(heatmap), window_size, stride=5, padding=window_size//2)
        heatmap_pooled = F.interpolate(heatmap_pooled.unsqueeze(0), 
                                       size=(heatmap[0].shape[0], 
                                             heatmap[0].shape[1]), mode='bilinear').squeeze(0)
        
        #  create binary mask
        heatmap_pooled[heatmap_pooled >= threshold] = 1
        heatmap_pooled[heatmap_pooled < threshold] = 0
        write_img(heatmap_pooled[0], name=f'heatmap_pooled.png')

        #  Find contour on heatmap_pooled
        heatmap_pooled = heatmap_pooled.cpu().numpy()
        #  Find contour on heatmap_pooled
        from skimage import measure
        contours = measure.find_contours(heatmap_pooled[0], 0.5)
        
        # DRAW CONTOUR ON HEATMAP AND SAVE IMAGE, DO NOT SHOW
        fig, ax = plt.subplots()
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='black')  # Assuming contours is a list of (x, y) points
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig('contour.png')
        plt.close(fig)  # Close the figure to prevent displaying it


        # In each contour, get the center pixel, other pixels map to 0
        new_heatmap = np.zeros_like(heatmap_pooled)
        for contour in contours:
            center = np.mean(contour, axis=0)
            center = np.round(center).astype(int)
            new_heatmap[0][center[0], center[1]] = 1
  


        pcls = []
        for i, h in enumerate(new_heatmap):
            h = h

            write_img(h, name=f'heatmap_{i}_pooled.png')
        
            x_centers = (x_edges_list[i][:-1] + x_edges_list[i][1:]) / 2
            y_centers = (y_edges_list[i][:-1] + y_edges_list[i][1:]) / 2

            # Generate a grid of indices
            x_indices, y_indices = np.indices(h.shape)

            # Repeat each index according to the corresponding histogram value
            x_repeat = np.repeat(x_indices, h.astype(int).flatten())
            y_repeat = np.repeat(y_indices, h.astype(int).flatten())
            # z_repeat = np.repeat(h.flatten(), h.astype(int).flatten())
            z_val = np.mean(h)
            z_repeat = np.full_like(x_repeat, z_val, dtype=float)

            # Map the indices to the bin centers to get the point coordinates
            pcl = np.column_stack((x_centers[x_repeat], y_centers[y_repeat], z_repeat))

            # num_levels = 5
            # pcl = auto_quantize_point_cloud(pcl, num_levels)

            # pcl = np.unique(pcl, axis=0)

            pcls.append(pcl)

        for batch_i, p in enumerate(batch_inputs_dict['points']):
            p = p.cpu().numpy()
            # # Nearest neighbors from p to pcls
            # from sklearn.neighbors import NearestNeighbors
            # nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(p[:,:2][valid])
            # distances, indices = nbrs.kneighbors(pcls[batch_i][:,:2])

            # draw_point(p, indices, name='pcl_img.png')
            draw_point_with(p[valid], name='pcl_thing_img.png', pcl2=pcls[batch_i], s2=20)

        print("none")

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
    pts = batch_inputs_dict['points']
    draw_point(pts[0].cpu().numpy(), name=f'pcl_gt_img_{time}.png')
    draw_point(pts[0].cpu().numpy()[valid], name=f'pcl_thing_img_{time}.png')

    pts[0] = pts[0][:,:3]

    embedding = [offset + xyz for offset, xyz in zip(pred_offsets, pts)]
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

def draw_point_with(pcl, indices=None, name='pcl_img.png', s=1, pcl2=None, s2=1):
    # Assuming `embedding` is your point cloud and `indices` are the indices of the centroids
    embedding_np = pcl  # Convert to numpy array for plotting
    
    # Create a new color array
    colors = np.full(embedding_np.shape[0], 'w')  # All points are blue
    
    # Create a new size array
    sizes = np.full(embedding_np.shape[0], s)  # All points are size 1
    
    if indices is not None:
        indices_np = indices.flatten()  # Flatten the indices array for indexing
        colors[indices_np] = 'b'  # Centroid points are red
        sizes[indices_np] = 10  # Centroid points are size 20

    # Create a 3D plot
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot()

    

    # Plot the point cloud in 3D
    scatter = ax.scatter(embedding_np[:, 0], embedding_np[:, 1], c=colors, s=sizes)
    if pcl2 is not None:
        scatter = ax.scatter(pcl2[:, 0], pcl2[:, 1], c='r', s=s2)

    # Set background color to black
    ax.set_facecolor('black')
    ax.set_aspect('auto')
    # ax.set_axis_off()

    # Hide grid
    # ax.grid(False)

    # Save the plot to an image file
    plt.savefig(name)


def auto_quantize_point_cloud(points, num_levels):
    """
    Automatically quantize a point cloud based on the number of quantization levels.
    
    Parameters:
    points (np.ndarray): The point cloud data as an (N, 3) array.
    num_levels (int): The number of quantization levels per dimension.
    
    Returns:
    np.ndarray: The quantized point cloud as an (N, 3) array.
    """
    # Determine the bounding box of the point cloud
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # Compute the range of the point cloud in each dimension
    range_coords = max_coords - min_coords
    
    # Compute the grid size
    grid_size = range_coords / num_levels
    
    # Quantize the points
    quantized_points = np.floor((points - min_coords) / grid_size + 0.5).astype(np.int32)
    
    # Reconstruct the quantized points
    quantized_points = quantized_points * grid_size + min_coords
    
    return quantized_points