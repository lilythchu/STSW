import torch
import torch.nn.functional as F

class STSWD():
    def __init__(self, ntrees=200, nlines=5, p=2, delta=2, device="cuda"):
        """
        Class for computing the TW distance between two point clouds
        Args:
            ntrees: Number of trees
            nlines: Number of lines per tree
            p: level of the norm
            delta: negative inverse of softmax temperature for distance based mass division
            device: device to run the code, follow torch convention
        """
        self.ntrees = ntrees
        self.device = device
        self.nlines = nlines
        self.p = p
        self.delta = delta
        self.eps = 1e-6

    def __call__(self, X, Y, root, intercept):
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Get mass
        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm and M == N
        
        combined_axis_coordinate, mass_X, mass_Y = self.get_mass_and_coordinate(X, Y, root, intercept)
        stw = self.stw_concurrent_lines(mass_X, mass_Y, combined_axis_coordinate)[0]

        return stw

    def stw_concurrent_lines(self, mass_X, mass_Y, combined_axis_coordinate):
        """
        Args:
            mass_X: (num_trees, num_lines, 2 * num_points)
            mass_Y: (num_trees, num_lines, 2 * num_points)
            combined_axis_coordinate: (num_trees, 2 * num_points)
        """
        coord_sorted, indices = torch.sort(combined_axis_coordinate, dim=-1)
        num_trees, num_lines = mass_X.shape[0], mass_X.shape[1]
        indices = indices.unsqueeze(1).repeat(1, num_lines, 1)

        # generate the cumulative sum of mass
        mass_X_sorted = torch.gather(mass_X, 2, indices)
        mass_Y_sorted = torch.gather(mass_Y, 2, indices)
        sub_mass = mass_X_sorted - mass_Y_sorted
        sub_mass_cumsum = torch.cumsum(sub_mass, dim=-1)
        sub_mass_target_cumsum = sub_mass + torch.sum(sub_mass, dim=-1, keepdim=True) - sub_mass_cumsum #(ntrees, nlines, 2*npoints)

        ### compute edge length
        edge_length = torch.diff(coord_sorted, prepend=torch.zeros((num_trees, 1), device=coord_sorted.device), dim=-1)
        edge_length = edge_length.unsqueeze(1) #(ntrees, 1, 2*npoints)

        # compute TW distance
        subtract_mass = (torch.abs(sub_mass_target_cumsum) ** self.p) * edge_length
        subtract_mass_sum = torch.sum(subtract_mass, dim=[-1,-2])
        tw = torch.mean(subtract_mass_sum) ** (1/self.p)

        return tw, sub_mass_target_cumsum, edge_length


    def get_mass_and_coordinate(self, X, Y, root, intercept):
        # for the last dimension
        # 0, 1, 2, ...., N -1 are from distribution 1
        # N, N + 1, ...., 2N -1 are from distribution 2
        N, dn = X.shape
        mass_X, axis_coordinate_X = self.project(X, root=root, intercept=intercept)
        mass_Y, axis_coordinate_Y = self.project(Y, root=root, intercept=intercept)
        mass_X = torch.cat((mass_X, torch.zeros((mass_X.shape[0], mass_X.shape[1], N), device=self.device)), dim=2)
        mass_Y = torch.cat((torch.zeros((mass_Y.shape[0], mass_Y.shape[1], N), device=self.device), mass_Y), dim=2)

        combined_axis_coordinate = torch.cat((axis_coordinate_X, axis_coordinate_Y), dim=-1)

        return combined_axis_coordinate, mass_X, mass_Y

    def project(self, input, root, intercept):
        """
        Args:
            input: (N, d)
            root: (ntrees, 1, d)
            intercept: (ntrees, nlines, d)
        
        Returns:
            mass_input: (ntrees, nlines, N)
            axis_coordinate: (ntrees, N)
        """
        N = input.shape[0]
        ntrees, nlines, d = intercept.shape
        # project input on great circle.
        input_alpha = root @ input.T #(ntrees, 1, N)
        input_pc = input - input_alpha.transpose(1, 2) @ root #(ntrees, N, d)
        input_pc = F.normalize(input_pc, p=2, dim=-1)
         
        ## get axis_coordinate 
        # coord based on distance from root to projections
        root_input_cosine = (root @ input.T).squeeze(1) #(ntrees, N) coordinate in vector root.
        axis_coordinate = torch.acos(torch.clamp(root_input_cosine, -1 + self.eps, 1 - self.eps)) #(ntrees, N)
        
        ## divide mass
        dist_cosine = intercept @ input_pc.transpose(1, 2) #(ntrees, nlines, N)
        dist = torch.acos(torch.clamp(dist_cosine, -1 + self.eps, 1 - self.eps)) 
        scale = torch.sin(axis_coordinate).unsqueeze(1) # (ntrees, 1, N)
        dist = dist * scale
        weight = -self.delta*dist #(ntrees, nlines, N)
        mass_input = torch.softmax(weight, dim=-2)/N

        return mass_input, axis_coordinate

def generate_spherical_trees_frames(ntrees, nlines, d, device='cuda'):    
    root = torch.randn(ntrees, 1, d, device=device)
    root = root / torch.norm(root, dim=-1, keepdim=True)
    intercept = torch.randn(ntrees, nlines, d, device=device)
    intercept_proj = intercept @ root.transpose(1, 2) #(ntrees, nlines, 1)
    intercept = intercept - intercept_proj @ root #(ntrees, nlines, d)
    intercept = F.normalize(intercept, p=2, dim=-1)

    return root, intercept

def unif_hypersphere(shape, device):
    samples = torch.randn(shape, device=device)
    samples = F.normalize(samples, p=2, dim=-1)
    return samples

def stswd(X, Y, ntrees=250, nlines=4, p=2, delta=2, device='cuda'):
    dn = X.shape[-1]
    TW_obj = STSWD(ntrees=ntrees, nlines=nlines, p=p, delta=delta, device=device)
    root, intercept = generate_spherical_trees_frames(ntrees, nlines, dn, device=device)
    stswd = TW_obj(X, Y, root, intercept)
    return stswd

def stswd_unif(X, ntrees=250, nlines=4, p=2, delta=2, device='cuda'):
    Y_unif = unif_hypersphere(X.shape, device=X.device) 
    stswd_unif = stswd(X, Y_unif, ntrees, nlines, p, delta, device)
    return stswd_unif
