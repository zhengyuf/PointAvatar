import torch
import torch.nn as nn

class PointCloud(nn.Module):
    def __init__(
        self,
        n_init_points,
        max_points=131072,
        init_radius=0.5,
        radius_factor=0.3
    ):
        super(PointCloud, self).__init__()
        self.radius_factor = radius_factor
        self.max_points = max_points
        self.init_radius = init_radius
        self.init(n_init_points)

    def init(self, n_init_points):
        print("current point number: ", n_init_points)
        # initialize sphere
        init_points = torch.rand(n_init_points, 3) * 2.0 - 1.0
        init_normals = nn.functional.normalize(init_points, dim=1)
        init_points = init_normals * self.init_radius
        self.register_parameter("points", nn.Parameter(init_points))

    def prune(self, visible_points):
        """Prune not rendered points"""
        self.points = nn.Parameter(self.points.data[visible_points])
        print(
            "Pruning points, original: {}, new: {}".format(
                len(visible_points), sum(visible_points)
            )
        )

    def upsample_points(self, new_points):
        self.points = nn.Parameter(torch.cat([self.points, new_points], dim=0))
