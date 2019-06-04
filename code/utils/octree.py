
import pycuda
import pycuda.autoinit
import pycuda.driver as cuda
import torch
_ = torch.cuda.FloatTensor(8) # somehow required for correct pytorch-pycuda interfacing, assuring they use the same context
from pycuda.compiler import SourceModule
import numpy as np
from utils.timer import Timer

class Octree:
    def __init__(self, points, centers, extents, parents, children):
        self.points = points
        self.centers = centers
        self.extents = extents
        self.parents = parents
        self.children = children


class TensorHolder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super().__init__()
        self.t = t
        self.gpudata = t.data_ptr()
    def get_pointer(self):
        return self.t.data_ptr()

kernel_code_generate_octree = '''
    #include <stdio.h>


    __global__ void generate_octree_shuffling(
        long *shuffled_vals,
        float *points,
        int N
    ){
        for(int n = threadIdx.x + blockDim.x*blockIdx.x; n < N; n += blockDim.x*gridDim.x) {
            float locx = points[0*N + n];
            float locy = points[1*N + n];
            float locz = points[2*N + n];
            long shuffled_val = 0;

            float scale = 0.5f;

            for(int level = 0; level < 20; level++) {
                bool locx_rule = locx > scale;
                bool locy_rule = locy > scale;
                bool locz_rule = locz > scale;

                shuffled_val = 8*shuffled_val + 4*locx_rule + 2*locy_rule + locz_rule;

                locx = locx - locx_rule * scale;
                locy = locy - locy_rule * scale;
                locz = locz - locz_rule * scale;

                scale = scale / 2;
            }
            shuffled_vals[n] = shuffled_val;
        }
    }

    __global__ void generate_octree_sidelengths(
        float *levels,
        float *centers,
        float *points,
        int *order,
        int N
    ){
        float locx0, locy0, locz0, locx1, locy1, locz1;
        float cx, cy, cz;
        float scale;
        bool locx_rule0, locx_rule1, locy_rule0, locy_rule1, locz_rule0, locz_rule1;
        for(int n = threadIdx.x + blockDim.x*blockIdx.x; n < N - 1; n += blockDim.x*gridDim.x) {
            locx0 = points[0*N + order[n]];
            locy0 = points[1*N + order[n]];
            locz0 = points[2*N + order[n]];
            locx1 = points[0*N + order[n+1]];
            locy1 = points[1*N + order[n+1]];
            locz1 = points[2*N + order[n+1]];

            cx = 0.5f;
            cy = 0.5f;
            cz = 0.5f;
            scale = 0.5f;

            for(int level = 0; level < 20; level++) {
                locx_rule0 = locx0 > scale;
                locy_rule0 = locy0 > scale;
                locz_rule0 = locz0 > scale;
                locx_rule1 = locx1 > scale;
                locy_rule1 = locy1 > scale;
                locz_rule1 = locz1 > scale;

                locx0 = locx0 - locx_rule0*scale;
                locy0 = locy0 - locy_rule0*scale;
                locz0 = locz0 - locz_rule0*scale;
                locx1 = locx1 - locx_rule1*scale;
                locy1 = locy1 - locy_rule1*scale;
                locz1 = locz1 - locz_rule1*scale;

                if(locx_rule0 != locx_rule1 || locy_rule0 != locy_rule1 || locz_rule0 != locz_rule1) {
                    levels[n] = level;
                    centers[0*(N-1) + n] = cx;
                    centers[1*(N-1) + n] = cy;
                    centers[2*(N-1) + n] = cz;
                    break;
                }

                cx = cx + locx_rule0 * scale - scale/2;
                cy = cy + locy_rule0 * scale - scale/2;
                cz = cz + locz_rule0 * scale - scale/2;

                scale = scale / 2;
            }
        }
    }

    __global__ void generate_octree_parents(
        float *levels,
        int *parents,
        int *nodes_needed,
        int N
    ) {
        bool found_left, found_equal_left, found_right;
        int left_idx, right_idx;
        float this_size, this_leftsize, left_size, right_size;
        for(int n = threadIdx.x + blockDim.x*blockIdx.x; n < N-1; n += blockDim.x*gridDim.x) {
            this_size = levels[n];
            if(this_size == 0) {
                found_left = 0;
                for(int leftlook = 1; leftlook <= n; leftlook++) {
                    this_leftsize = levels[n - leftlook];
                    if(this_leftsize == 0) {
                        found_left = 1;
                        break;
                    }
                }

                if(!found_left) {
                    parents[n] = -1;
                }
            }
            else if(this_size > 0){
                found_left = 0;
                found_equal_left = 0;
                found_right = 0;
                right_size = 2.0;

                for(int leftlook = 1; leftlook <= n; leftlook++) {
                    this_leftsize = levels[n - leftlook];
                    if(this_leftsize < this_size && this_leftsize >= 0) {
                        if(!found_left) {
                            // this is the first candidate parent we meet
                            found_left = 1;
                            left_idx = n - leftlook;
                            left_size = this_leftsize;
                        }
                        else {
                            if(left_size == this_leftsize) {
                                // we found a more left parent
                                left_idx = n - leftlook;
                            }
                            else if(left_size > this_leftsize) {
                                // we have finished our search
                                break;
                            }
                        }
                    }
                    else if (this_leftsize == this_size && !found_left) {
                        found_equal_left = 1;
                        break;
                    }
                }
                if(found_equal_left) continue;
                for(int rightlook = 1; rightlook < N - 1 - n; rightlook ++) {
                    right_size = levels[n + rightlook];
                    if(right_size < this_size && right_size >= 0) {
                        right_idx = n + rightlook;
                        found_right = 1;
                        break;
                    }
                }
                if(found_left) {
                    if(found_right && right_size > left_size) {
                        parents[n] = right_idx;
                        nodes_needed[n] = 1;
                    }
                    else {
                        parents[n] = left_idx;
                        nodes_needed[n] = 1;
                    }
                }
                else {
                    if(found_right) {
                        parents[n] = right_idx;
                        nodes_needed[n] = 1;
                    }
                }
            }
        }
    }

    __global__ void generate_octree_treefiller(
        int *nodes_needed,
        int *node_indices,
        float *centers_in,
        float *centers_out,
        float *levels_in,
        int *levels_out,
        int *parents_in,
        int *parents_out,
        int *children,
        int N,
        int Nout
    ) {
        for(int n = threadIdx.x + blockDim.x*blockIdx.x; n < N - 1; n += blockDim.x*gridDim.x) {
            if(nodes_needed[n]) {
                // straightforward: put our center and extent at the correct location, and put our pointer in our parent's correct slot
                int our_index = node_indices[n];
                float our_center_x = centers_in[0*(N-1) + n];
                float our_center_y = centers_in[1*(N-1) + n];
                float our_center_z = centers_in[2*(N-1) + n];
                int our_level = (int) levels_in[n];
                int parent_index;
                float parent_center_x, parent_center_y, parent_center_z;
                if(parents_in[parents_in[n]] == -1) {
                    parent_index = 0;
                    parent_center_x = 0.5f;
                    parent_center_y = 0.5f;
                    parent_center_z = 0.5f;
                }
                else {
                    parent_index = node_indices[parents_in[n]];
                    parent_center_x = centers_in[0*(N-1) + parents_in[n]];
                    parent_center_y = centers_in[1*(N-1) + parents_in[n]];
                    parent_center_z = centers_in[2*(N-1) + parents_in[n]];
                }

                int child_index = (parent_center_x < our_center_x) + (parent_center_y < our_center_y) * 2 + (parent_center_z < our_center_z) * 4;
                children[child_index * Nout + parent_index] = our_index;

                centers_out[0 * Nout + our_index] = our_center_x;
                centers_out[1 * Nout + our_index] = our_center_y;
                centers_out[2 * Nout + our_index] = our_center_z;
                levels_out[our_index] = our_level;
                parents_out[our_index] = parent_index;
            }
        }
    }

    __global__ void generate_octree_populater(
        float *points,
        int *children,
        float *centers_out,
        int N,
        int Nout
    ) {
        for(int n = threadIdx.x + blockDim.x*blockIdx.x; n < N; n += blockDim.x*gridDim.x) {
            float thisloc_x = points[0*N + n];
            float thisloc_y = points[1*N + n];
            float thisloc_z = points[2*N + n];

            int node_index = 0;
            float parent_x = 0.5f;
            float parent_y = 0.5f;
            float parent_z = 0.5f;
            int child_index = (thisloc_x > parent_x) + 2*(thisloc_y > parent_y) + 4*(thisloc_z > parent_z);
            while(children[child_index * Nout + node_index] > 0) {
                node_index = children[child_index * Nout + node_index];
                parent_x = centers_out[0 * Nout + node_index];
                parent_y = centers_out[1 * Nout + node_index];
                parent_z = centers_out[2 * Nout + node_index];
                child_index = (thisloc_x > parent_x) + 2*(thisloc_y > parent_y) + 4*(thisloc_z > parent_z);
            }
            // negative values denote elements of the point set, positive elements denote octree nodes
            children[child_index * Nout + node_index] = - 1 - n;
        }
    }
'''

module_generate_octree = SourceModule(kernel_code_generate_octree)
generate_octree_kernel_shuffling = module_generate_octree.get_function('generate_octree_shuffling')
generate_octree_kernel_sidelengths = module_generate_octree.get_function('generate_octree_sidelengths')
generate_octree_kernel_parents = module_generate_octree.get_function('generate_octree_parents')
generate_octree_kernel_treefiller = module_generate_octree.get_function('generate_octree_treefiller')
generate_octree_kernel_populater = module_generate_octree.get_function('generate_octree_populater')

def generate_octree(points):
    N = points.shape[1]

    # step 1: convert the locations to shuffled fixed-point floats, represented in an int
    shuffled_values = points.new_zeros(N, dtype=torch.int64)
    scaled_points = points.clone()
    mins = [0,0,0]
    maxs = [0,0,0]
    for c in range(3):
        mins[c] = scaled_points[c].min()
        maxs[c] = scaled_points[c].max()
        scaled_points[c] = (scaled_points[c] - mins[c]) / (maxs[c] - mins[c])

    generate_octree_kernel_shuffling(
        TensorHolder(shuffled_values),
        TensorHolder(scaled_points), np.int32(N),
        grid=(int(60), int(1), int(1)), block=(int(1024), int(1), int(1))
    )

    # step 2: sort the shuffled values
    order = torch.from_numpy(np.argsort(shuffled_values.cpu().numpy(), kind='stable').astype(np.int32)).cuda()

    # step 3: compute the derived squares, and denote the side length
    levels = points.new_zeros(N - 1) - 1
    centers = points.new_zeros((3, N - 1))

    generate_octree_kernel_sidelengths(
        TensorHolder(levels), TensorHolder(centers),
        TensorHolder(scaled_points), TensorHolder(order),
        np.int32(N),
        grid=(int(60), int(1), int(1)), block=(int(1024), int(1), int(1))
    )

    # step 4: find the node parents. Every internal node is present between one and seven times in this vector
    #  [each point knows its parent by the smallest sidelength square derived from it]
    #  each node knows its parent by the smallest sidelength square to the left or right of it in the vector
    parents = points.new_zeros(N-1, dtype=torch.int32) - 2
    nodes_needed = points.new_zeros(N-1, dtype=torch.int32)

    generate_octree_kernel_parents(
        TensorHolder(levels), TensorHolder(parents),
        TensorHolder(nodes_needed), np.int32(N),
        grid=(int(60), int(1), int(1)), block=(int(1024), int(1), int(1))
    )

    # step 5: construct the tree out of the vector of parent-relationships
    nr_nodes = nodes_needed.sum().item() + 1
    children = points.new_zeros((8, nr_nodes), dtype=torch.int32)
    centers_out = points.new_zeros((3, nr_nodes))
    centers_out[:, 0] = 0.5
    levels_out = points.new_zeros(nr_nodes, dtype=torch.int32)
    levels_out[0] = 0
    parents_out = points.new_zeros(nr_nodes, dtype=torch.int32)
    parents_out[0] = -1
    node_indices = nodes_needed.cumsum(dim=0, dtype=torch.int32)

    generate_octree_kernel_treefiller(
        TensorHolder(nodes_needed), TensorHolder(node_indices),
        TensorHolder(centers), TensorHolder(centers_out),
        TensorHolder(levels), TensorHolder(levels_out), 
        TensorHolder(parents), TensorHolder(parents_out),
        TensorHolder(children),
        np.int32(N), np.int32(nr_nodes),
        grid=(int(60), int(1), int(1)), block=(int(1024), int(1), int(1))
    )

    # step 6: populate the tree with the actual points
    empty_children_pre = (children == 0).sum()
    generate_octree_kernel_populater(
        TensorHolder(scaled_points), TensorHolder(children),
        TensorHolder(centers_out),
        np.int32(N), np.int32(nr_nodes),
        grid=(int(60), int(1), int(1)), block=(int(1024), int(1), int(1))
    )
    empty_children_post = (children == 0).sum()

    lost_points = empty_children_post - (empty_children_pre - N)
    if lost_points > 0:
        print("[OCTREE] WARNING> point collision: %d points were lost." % lost_points)

    torch.cuda.synchronize()

    extents_out = torch.ones(3,1).cuda() * levels_out[None].float()
    extents_out = 1 / 2 ** (extents_out + 1)

    for c in range(3):
        centers_out[c] = centers_out[c] * (maxs[c] - mins[c]) + mins[c]
        extents_out[c] = extents_out[c] * (maxs[c] - mins[c])

    return Octree(points, centers_out, extents_out, parents_out, children)

kernel_code_crawl_octree = '''
    #include <stdio.h>

    __global__ void crawl_octree_NN(
        float *points_from,
        float *points_tree,
        float *node_centers,
        int *node_children,
        float *node_extents,
        int *node_parents,
        float *ndists,
        int *nns,
        int nr_points_from,
        int nr_points_tree,
        int nr_nodes,
        int own_tree
    ){
        for(int n = threadIdx.x + blockDim.x*blockIdx.x; n < nr_points_from; n += blockDim.x*gridDim.x) {
            int nn = -1;
            float ndist = 0.0f;

            // step 1: search for the direct parent
            float thisloc_x = points_from[0*nr_points_from + n];
            float thisloc_y = points_from[1*nr_points_from + n];
            float thisloc_z = points_from[2*nr_points_from + n];

            int parent_index = 0;
            float parent_x = node_centers[0*nr_nodes + 0];
            float parent_y = node_centers[1*nr_nodes + 0];
            float parent_z = node_centers[2*nr_nodes + 0];
            int child_index = (thisloc_x > parent_x) + 2*(thisloc_y > parent_y) + 4*(thisloc_z > parent_z);
            while(node_children[child_index * nr_nodes + parent_index] > 0) {
                parent_index = node_children[child_index * nr_nodes + parent_index];
                parent_x = node_centers[0 * nr_nodes + parent_index];
                parent_y = node_centers[1 * nr_nodes + parent_index];
                parent_z = node_centers[2 * nr_nodes + parent_index];
                child_index = (thisloc_x > parent_x) + 2*(thisloc_y > parent_y) + 4*(thisloc_z > parent_z);
            }

            // step 2: due to the definition of octrees, this parent has at least one child not equal to this point
            int node_index = parent_index;
            for(int other_child = 0; other_child < 8; other_child++) {
                // we don't want to get ourself
                if(own_tree && other_child == child_index && node_index == parent_index) {
                    continue;
                }
                int other_node = node_children[other_child * nr_nodes + node_index];
                if(other_node > 0) {
                    // which may, however, be split up further
                    // then we just recursively search for an actual point child
                    node_index = other_node;
                    other_child = 0;
                }
                else if (other_node < 0) {
                    // we found an actual point!
                    nn = - (other_node + 1);
                    float thatloc_x = points_tree[0*nr_points_tree + nn];
                    float thatloc_y = points_tree[1*nr_points_tree + nn];
                    float thatloc_z = points_tree[2*nr_points_tree + nn];
                    ndist = (thisloc_x - thatloc_x) * (thisloc_x - thatloc_x);
                    ndist+= (thisloc_y - thatloc_y) * (thisloc_y - thatloc_y);
                    ndist+= (thisloc_z - thatloc_z) * (thisloc_z - thatloc_z);
                }
            }

            // after that, perform a branch-and-bound strategy on the entire tree to find the closest neighbour (which may be on the other side of the parent boundary)
            node_index = 0;
            for(int child = 0; child < 8; child++) {
                int child_node = node_children[child * nr_nodes + node_index];
                if(child_node > 0) {
                    // bound the best distance in this child
                    float closest_x = thisloc_x;
                    closest_x = max(closest_x, node_centers[0 * nr_nodes + child_node] - node_extents[0 * nr_nodes + child_node]);
                    closest_x = min(closest_x, node_centers[0 * nr_nodes + child_node] + node_extents[0 * nr_nodes + child_node]);
                    float closest_y = thisloc_y;
                    closest_y = max(closest_y, node_centers[1 * nr_nodes + child_node] - node_extents[1 * nr_nodes + child_node]);
                    closest_y = min(closest_y, node_centers[1 * nr_nodes + child_node] + node_extents[1 * nr_nodes + child_node]);
                    float closest_z = thisloc_z;
                    closest_z = max(closest_z, node_centers[2 * nr_nodes + child_node] - node_extents[2 * nr_nodes + child_node]);
                    closest_z = min(closest_z, node_centers[2 * nr_nodes + child_node] + node_extents[2 * nr_nodes + child_node]);

                    float closest_dist = (closest_x - thisloc_x) * (closest_x - thisloc_x);
                    closest_dist += (closest_y - thisloc_y) * (closest_y - thisloc_y);
                    closest_dist += (closest_z - thisloc_z) * (closest_z - thisloc_z);
                    
                    // if it is lower than the current best, investigate this subtree recursively
                    if(closest_dist < ndist) {
                        node_index = child_node;
                        child = -1;
                    }
                }
                else if(child_node < 0) {
                    int child_point = - child_node - 1;
                    if(!(own_tree && child == child_index && node_index == parent_index)) {
                        // this is an actual point -- investigate the distance
                        float thatloc_x = points_tree[0*nr_points_tree + child_point];
                        float thatloc_y = points_tree[1*nr_points_tree + child_point];
                        float thatloc_z = points_tree[2*nr_points_tree + child_point];
                        float thatdist = (thisloc_x - thatloc_x) * (thisloc_x - thatloc_x);
                        thatdist += (thisloc_y - thatloc_y) * (thisloc_y - thatloc_y);
                        thatdist += (thisloc_z - thatloc_z) * (thisloc_z- thatloc_z);
                        if(thatdist < ndist) {
                            nn = child_point;
                            ndist = thatdist;
                        }
                    }
                    else if(child_point != n) {
                        // this is an actual point, it's not us and it's supposed to be
                        // we got lost when building the octree: duplicate points
                        ndist = -1;
                        nn = -1;
                        break;
                    }
                }

                while(child == 7 && node_parents[node_index] >= 0) {
                    // move one level back up 
                    child = 0;
                    int new_parent_index = node_parents[node_index];
                    // and continue with the next child
                    while(child < 8 && node_children[child * nr_nodes + new_parent_index] != node_index) {
                        child++;
                    }
                    node_index = new_parent_index;
                }
            }

            // finally, write everything to the output
            if(ndist > 0) ndist = sqrt(ndist);
            ndists[n] = ndist;
            nns[n] = nn;
        }
    }

    __global__ void crawl_octree_radius_count(
        float *points_tree,
        int *point_counts,
        float *node_centers,
        int *node_children,
        float *node_extents,
        int *node_parents,
        float radius,
        int nr_points_tree,
        int nr_nodes
    ){
        float radius2 = radius * radius;
        for(int n = threadIdx.x + blockDim.x*blockIdx.x; n < nr_points_tree; n += blockDim.x*gridDim.x) {
            // step 1: search for the direct parent
            float thisloc_x = points_tree[0*nr_points_tree + n];
            float thisloc_y = points_tree[1*nr_points_tree + n];
            float thisloc_z = points_tree[2*nr_points_tree + n];

            int parent_index = 0;
            float parent_x = node_centers[0*nr_nodes + 0];
            float parent_y = node_centers[1*nr_nodes + 0];
            float parent_z = node_centers[2*nr_nodes + 0];
            int child_index = (thisloc_x > parent_x) + 2*(thisloc_y > parent_y) + 4*(thisloc_z > parent_z);
            while(node_children[child_index * nr_nodes + parent_index] > 0) {
                parent_index = node_children[child_index * nr_nodes + parent_index];
                parent_x = node_centers[0 * nr_nodes + parent_index];
                parent_y = node_centers[1 * nr_nodes + parent_index];
                parent_z = node_centers[2 * nr_nodes + parent_index];
                child_index = (thisloc_x > parent_x) + 2*(thisloc_y > parent_y) + 4*(thisloc_z > parent_z);
            }

            int count = 0;

            // after that, perform a branch-and-bound strategy on the entire tree to find all the points within this radius
            int node_index = 0;
            for(int child = 0; child < 8; child++) {
                int child_node = node_children[child * nr_nodes + node_index];
                if(child_node > 0) {
                    // bound the best distance in this child
                    float closest_x = thisloc_x;
                    closest_x = max(closest_x, node_centers[0 * nr_nodes + child_node] - node_extents[0 * nr_nodes + child_node]);
                    closest_x = min(closest_x, node_centers[0 * nr_nodes + child_node] + node_extents[0 * nr_nodes + child_node]);
                    float closest_y = thisloc_y;
                    closest_y = max(closest_y, node_centers[1 * nr_nodes + child_node] - node_extents[1 * nr_nodes + child_node]);
                    closest_y = min(closest_y, node_centers[1 * nr_nodes + child_node] + node_extents[1 * nr_nodes + child_node]);
                    float closest_z = thisloc_z;
                    closest_z = max(closest_z, node_centers[2 * nr_nodes + child_node] - node_extents[2 * nr_nodes + child_node]);
                    closest_z = min(closest_z, node_centers[2 * nr_nodes + child_node] + node_extents[2 * nr_nodes + child_node]);

                    float closest_dist = (closest_x - thisloc_x) * (closest_x - thisloc_x);
                    closest_dist += (closest_y - thisloc_y) * (closest_y - thisloc_y);
                    closest_dist += (closest_z - thisloc_z) * (closest_z - thisloc_z);
                    
                    // if it is lower than the current best, investigate this subtree recursively
                    if(closest_dist < radius) {
                        node_index = child_node;
                        child = -1;
                    }
                }
                else if(child_node < 0) {
                    int child_point = - child_node - 1;
                    if(!(child == child_index && node_index == parent_index)) {
                        // this is an actual point, and it's not us -- investigate the distance
                        float thatloc_x = points_tree[0*nr_points_tree + child_point];
                        float thatloc_y = points_tree[1*nr_points_tree + child_point];
                        float thatloc_z = points_tree[2*nr_points_tree + child_point];
                        float thatdist = (thisloc_x - thatloc_x) * (thisloc_x - thatloc_x);
                        thatdist += (thisloc_y - thatloc_y) * (thisloc_y - thatloc_y);
                        thatdist += (thisloc_z - thatloc_z) * (thisloc_z- thatloc_z);
                        if(thatdist < radius2) {
                            count++;
                        }
                    }
                    else if(child_point != n)
                    {
                        // this is an actual point, it's not us and it's supposed to be
                        // we got lost when building the octree: duplicate points
                        count = -1;
                        break;
                    }
                }

                while(child == 7 && node_parents[node_index] >= 0) {
                    // move one level back up 
                    child = 0;
                    int new_parent_index = node_parents[node_index];
                    // and continue with the next child
                    while(child < 8 && node_children[child * nr_nodes + new_parent_index] != node_index) {
                        child++;
                    }
                    node_index = new_parent_index;
                }
            }

            // finally, write everything to the output
            point_counts[n] = count;
        }
    }
'''

module_crawl_octree = SourceModule(kernel_code_crawl_octree)
craw_octree_kernel_NN = module_crawl_octree.get_function('crawl_octree_NN')
crawl_octree_kernel_radius_count = module_crawl_octree.get_function('crawl_octree_radius_count')

octree_debug = False

def chamfer(points_from, tree, own_tree=False):
    """
    Calculate the minimum distance from each of the points in points_from to the points in the octree.
    points_from (3 x N_from torch.float32 cuda array)
    tree (Octree)
    own_tree=False -- If the tree was built from points_from, then the points should not count themselves for this distance calculation
                   -- This is not the case for 99+% of this function's usage
    """
    N_from = points_from.shape[1]
    N_tree = tree.points.shape[1]
    nr_nodes = tree.children.shape[1]

    ndists = points_from.new_zeros(N_from, dtype=torch.float32)
    nns = points_from.new_zeros(N_from, dtype=torch.int32)

    craw_octree_kernel_NN(
        TensorHolder(points_from), TensorHolder(tree.points),
        TensorHolder(tree.centers), TensorHolder(tree.children), TensorHolder(tree.extents), TensorHolder(tree.parents),
        TensorHolder(ndists), TensorHolder(nns), 
        np.int32(N_from), np.int32(N_tree),
        np.int32(nr_nodes), np.int32(own_tree),
        grid=(int(60), int(1), int(1)), block=(int(1024), int(1), int(1))
    )

    nr_lost = (ndists < 0).sum().item()
    if nr_lost > 0:
        print("[OCTREE|WARNING] %d points were lost, as observed by chamfer()" % nr_lost)

    # if octree_debug and max(N_from, N_tree) < 8*1024 and nr_lost == 0:
    #     # DEBUG test
    #     print("[OCTREE|DEBUG asserts running]")
    #     all_distances = np.sqrt(np.power((points_from[:,None,:] - tree.points[:,:,None]), 2).sum(0, keepdim=False)).cuda()
    #     if own_tree:
    #         for c in range(N_from):
    #             all_distances[c,c] = np.inf
    #     real_dists, real_nns = all_distances.min(1)
    #     assert((torch.abs(real_dists - ndists) / real_dists < 1e-5).all())
    #     assert((real_nns.int() == nns).all())

    torch.cuda.synchronize()

    return ndists

def radius_count(tree, radius):
    """
    Calculate the number of points within a certain radius of one another within a pre-built tree
    tree (Octree)
    radius (scalar)
    """
    N_tree = tree.points.shape[1]
    nr_nodes = tree.children.shape[1]

    counts = points.new_zeros(N_tree, dtype=torch.int32)

    crawl_octree_kernel_radius_count(
        TensorHolder(tree.points), TensorHolder(counts),
        TensorHolder(tree.centers), TensorHolder(tree.children), TensorHolder(tree.extents), TensorHolder(tree.parents),
        np.float32(radius), np.int32(N_tree), np.int32(nr_nodes),
        grid=(int(60), int(1), int(1)), block=(int(1024), int(1), int(1))
    )

    nr_lost = sum(counts < 0)
    if nr_lost > 0:
        print("[OCTREE|WARNING] %d points were lost, as observed by radius_count()" % nr_lost)

    # if octree_debug and N_tree < 8*1024 and nr_lost == 0:
    #     # DEBUG test
    #     all_distances = np.sqrt(np.power((points[:,None,:] - points[:,:,None]), 2).sum(0, keepdim=False)).cuda()
    #     for c in range(N_tree):
    #         all_distances[c,c] = np.inf
    #     real_counts = (all_distances < radius).sum(1).int()
    #     assert(torch.all(counts == real_counts))

    torch.cuda.synchronize()

    return counts


if __name__ == "__main__":
    N_points = 1024*16*16
    cube_edge = 10
    points = np.random.rand(3, N_points) * cube_edge - cube_edge / 3
    points = torch.from_numpy(points.astype(np.float32)).cuda()

    points[:, 1] = points[:, 0]

    tree = generate_octree(points)

    with Timer(message="Octree creation"):
        tree = generate_octree(points)

    import sys
    sys.exit(0)

    with Timer(message="Chamfer calculation"):
        chamfer(points, tree, own_tree=True)

    # expected value here: 
    count_radius = 0.5
    expected_neighbours_per_point = 4 / 3 * np.pi * count_radius ** 3 * N_points / cube_edge ** 3
    print("Radius count: expected roughly %f neighbours per point" % expected_neighbours_per_point)
    with Timer(message="Radius count"): 
        point_counts = radius_count(tree, radius=count_radius)
    print("Radius count: got %f neighbours per point" % point_counts.float().mean())

    point_counts = radius_count(tree, radius=count_radius)
