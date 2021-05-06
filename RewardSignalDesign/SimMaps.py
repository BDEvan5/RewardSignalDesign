import numpy as np 
from scipy import ndimage
from matplotlib import pyplot as plt
import yaml
import csv
from PIL import Image

import LibFunctions as lib
from TrajectoryPlanner import MinCurvatureTrajectory, ObsAvoidTraj, ShortestTraj, Max_velocity

class SimMap:
    def __init__(self, config) -> None:
        self.config = config
        self.map_name = config['map']['name']

        self.map_img = None
        self.obs_img = None

        self.start = np.array([0, 0])
        self.origin = None
        self.wpts = None
        self.vs = None

        self.height = None
        self.width = None
        self.resolution = None

        self.diffs = None
        self.l2s = None

        self.targets = []

        self.load_map()

    def load_map(self):
        # load yaml
        file_name = 'maps/' + self.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = documents.items()

        self.yaml_file = dict(yaml_file)
        self.resolution = self.yaml_file['resolution']
        self.origin = self.yaml_file['origin']

        # load traj
        track_data = []
        filename = 'maps/' + self.map_name + '_opti.csv'
        
        try:
            with open(filename, 'r') as csvfile:
                csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
                for lines in csvFile:  
                    track_data.append(lines)
        except FileNotFoundError:
            raise FileNotFoundError("No map traj")

        track = np.array(track_data)
        print(f"Track Loaded: {filename} in env map")

        self.N = len(track)
        self.ss = track[:, 0]
        self.wpts = track[:, 1:3]
        self.vs = track[:, 5]
        self.thetas = track[:, 3]

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

        map_img_path = 'maps/' + self.yaml_file['image']
        np.array(Image.open(map_img_path))
        self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
        try:
            self.map_img = self.map_img[:, :, 0] / 255
        except:
            self.map_img = self.map_img / 255

        self.dt = ndimage.distance_transform_edt(self.map_img) 
        self.dt = np.array(self.dt *self.resolution)

        self.map_img = np.ones_like(self.map_img) - self.map_img

        self.obs_img = np.zeros_like(self.map_img)
        self.obs_img_plan = np.zeros_like(self.map_img)

        self.width = self.map_img.shape[1]
        self.height = self.map_img.shape[0]

    def calculate_traj(self):
        n_set = MinCurvatureTrajectory(self.track_pts, self.nvecs, self.ws)
        deviation = np.array([self.nvecs[:, 0] * n_set[:, 0], self.nvecs[:, 1] * n_set[:, 0]]).T
        self.wpts = self.track_pts + deviation

        # self.render_map(4, False)

        self.vs = Max_velocity(self.wpts, self.config, True)
        dss, ths = convert_pts_s_th(self.wpts)
        self.thetas = ths
        ss = np.cumsum(dss)
        ks = np.zeros_like(ths[:, None]) #TODO: add the curvature

        track = np.concatenate([ss[:, None], self.wpts[:-1], ths[:, None], ks, self.vs], axis=-1)

        filename = 'maps/' + self.map_name + '_opti.csv'

        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(track)

        print(f"Track Saved in File: {filename}")
        # plt.show()

    def get_reference_path(self):
        if self.wpts is None:
            self.load_map()
        
        # self.render_map(wait=True)

        return self.wpts, self.vs

    def load_center_path(self):
        if self.t_pts is None:
            track_data = []
            filename = 'maps/' + self.map_name + '_std.csv'
            
            try:
                with open(filename, 'r') as csvfile:
                    csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
            
                    for lines in csvFile:  
                        track_data.append(lines)
            except FileNotFoundError:
                raise FileNotFoundError("No map file center pts")

            track = np.array(track_data)
            print(f"Track Loaded: {filename} in env map")

            self.t_pts = track[:, 0:2]
        else:
            return 

    def get_center_path(self):
        return self.t_pts


    def get_optimal_path(self):
        raise NotImplementedError

    def render_map(self, figure_n=4, wait=False):
        f = plt.figure(figure_n)
        plt.clf()

        plt.xlim([0, self.width])
        plt.ylim([self.height, 0])

        cx, cy = self.convert_positions(self.wpts)
        plt.plot(cx, cy, '--', linewidth=1)

        # tx, ty = self.convert_positions(self.targets)
        # plt.plot(tx, ty, 'o')

        if self.obs_img is None:
            plt.imshow(self.map_img, origin='lower')
        else:
            plt.imshow(self.obs_img + self.map_img, origin='lower')

        plt.gca().set_aspect('equal', 'datalim')

        plt.pause(0.0001)
        if wait:
            plt.show()

    def xy_to_row_column(self, pt_xy):
        c = int((pt_xy[0] - self.origin[0]) / self.resolution)
        r = int((pt_xy[1] - self.origin[1]) / self.resolution)

        return c, r

    def check_scan_location(self, pt):
        c, r = self.xy_to_row_column(pt)
        if abs(c) > self.width -2 or abs(r) > self.height -2:
            return True
        val = self.dt[r, c]
        # plt.imshow(self.dt)
        # plt.show()
        if val < 0.1:
            return True
        if self.obs_img[r, c]:
            return True
        return False

    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.xy_to_row_column(pt)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    def reset_map(self, n=10):
        self.obs_img = np.zeros_like(self.obs_img)

        if n != 0:
            # obs_size = [self.width/600, self.height/600]
            obs_size = self.config['map']['obs_size']
            obs_size = [obs_size, obs_size]
            # obs_size = [0.3, 0.3]
            # obs_size = [1, 1]
            obs_size = np.array(obs_size) / self.resolution

            buffer = 4
            rands = np.random.randint(5, self.N-5, n)
            rands = np.sort(rands)
            diffs = rands[1:] - rands[:-1]
            diffs = np.insert(diffs, 0, buffer+1)
            rands = rands[diffs>buffer]

            n = len(rands)
            obs_locs = []
            for i in range(n):
                pt = self.wpts[rands[i]][:, None]
                obs_locs.append(pt[:, 0])

            for obs in obs_locs:
                for i in range(0, int(obs_size[0])):
                    for j in range(0, int(obs_size[1])):
                        x, y = self.xy_to_row_column([obs[0], obs[1]])
                        self.obs_img[y+j, x+i] = 1

        return self.get_reference_path()

    def find_nearest_pt(self, point):
        dots = np.empty((self.wpts.shape[0]-1, ))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - self.wpts[i, :]), self.diffs[i, :])
        t = dots / self.l2s

        t = np.clip(dots / self.l2s, 0.0, 1.0)
        projections = self.wpts[:-1,:] + (t*self.diffs.T).T
        dists = np.linalg.norm(point - projections, axis=1)

        min_dist_segment = np.argmin(dists)

        return min_dist_segment
  

class ForestMap:
    def __init__(self,config) -> None:
        self.config = config
        self.length = config['forest_len']
        self.width = config['forest_w']
        self.start = config['start']
        self.end = config['end']
        self.resoltuion = config['resolution']
        self.obs_size = config['obs_size']
        self.obs_buf = config['obs_buf']
        self.n_obs = config['n_obs']
        self.plan_scale = config['plan_scale']

        self.max_v = config['max_v']
        self.wpts = None

        self.h = int(self.length / self.resoltuion)
        self.w = int(self.width / self.resoltuion)
        self.obs_img = np.zeros((self.w, self.h))
        self.obs_img_plan = np.zeros((self.w, self.h))

    def xy_to_row_column(self, pt):
        c = int(round(np.clip(pt[0] / self.resoltuion, 0, self.w-2)))
        r = int(round(np.clip(pt[1] / self.resoltuion, 0, self.h-2)))
        return c, r

    def reset_static_map(self, n=None):
        if n is None:
            n = self.n_obs
        self.obs_img = np.zeros_like(self.obs_img)

        obs_size = np.array([self.obs_size, self.obs_size])
        x, y = self.xy_to_row_column(obs_size)
        norm_obs_size = [x, y]
        x, y = self.xy_to_row_column(obs_size * self.plan_scale)
        obs_size_plan = [x, y]

        y_end = self.length - self.obs_buf - self.obs_size
        tys = np.linspace(self.obs_buf, y_end, n)
        txs = np.random.uniform(0, self.width-self.obs_size, n)
        obs_locs = np.array([txs, tys]).T

        for obs in obs_locs:
            for i in range(0, norm_obs_size[0]):
                for j in range(0, norm_obs_size[1]):
                    x, y = self.xy_to_row_column([obs[0], obs[1]])
                    x = np.clip(x+i, 0, self.w-1)
                    y = np.clip(y+j, 0, self.h-1)
                    self.obs_img[x, y] = 1

        for obs in obs_locs:
            for i in range(0, obs_size_plan[0]):
                for j in range(0, obs_size_plan[1]):
                    x, y = self.xy_to_row_column([obs[0], obs[1]])
                    x = np.clip(x+i, 0, self.w-1)
                    y = np.clip(y+j, 0, self.h-1)
                    self.obs_img_plan[x, y] = 1

        return self.get_reference_path()

    def check_scan_location(self, x_in):
        if x_in[0] < 0 or x_in[1] < 0:
            return True
        if x_in[0] > self.width or x_in[1] > self.length:
            return True
        x, y = self.xy_to_row_column(x_in)
        if self.obs_img[x, y]:
            return True

    def get_reference_path(self):
        n_pts = self.length * 3
        tys = np.linspace(self.start[1], self.end[1], n_pts)
        txs = np.ones(n_pts) * self.width/2

        self.wpts = np.concatenate([txs[:, None], tys[:, None]], axis=-1)
        vs = np.ones(n_pts) * self.max_v

        return self.wpts, vs 

    def get_optimal_path(self):
        raise NotImplementedError

    def render_map(self, figure_n=1, wait=False):
        #TODO: draw the track boundaries nicely
        f = plt.figure(figure_n)
        plt.clf()

        plt.xlim([0, self.w])
        plt.ylim([self.h, 0])

        if self.wpts is not None:
            xs, ys = [], []
            for pt in self.wpts:
                x, y = self.xy_to_row_column(pt)
                # plt.plot(x, y, '+', markersize=14)
                xs.append(x)
                ys.append(y)
            plt.plot(xs, ys, '--', color='g', linewidth=2)

        plt.imshow(self.obs_img.T)

        # plt.gca().set_aspect('equal', 'datalim')
        x, y = self.xy_to_row_column(self.end)
        plt.plot(x, y, '*', markersize=14)        
        x, y = self.xy_to_row_column(self.start)
        plt.plot(x, y, '*', markersize=14)

        plt.pause(0.0001)
        if wait:
            plt.show()
            pass



def convert_pts_s_th(pts):
    N = len(pts)
    s_i = np.zeros(N-1)
    th_i = np.zeros(N-1)
    for i in range(N-1):
        s_i[i] = lib.get_distance(pts[i], pts[i+1])
        th_i[i] = lib.get_bearing(pts[i], pts[i+1])

    return s_i, th_i



class CarObs:
    def __init__(self, start_pos, velocity) -> None:
        self.vel = velocity
        self.pos = np.array(start_pos, dtype=np.float)

    def update_pos(self, dt):
        self.pos[1] += self.vel * dt
        

# Testing functions
def test_sim_map_obs():
    name = 'race_track'
    env_map = SimMap(name)
    env_map.reset_map()

    wpts = env_map.get_optimal_path()
    env_map.render_map(wait=True)

def test_forest_map_obs():
    name = 'forest'
    env_map = ForestMap(name)
    env_map.reset_map()

    wpts = env_map.get_optimal_path()
    env_map.render_map(wait=True)



if __name__ == "__main__":

    # test_sim_map_obs()
    test_forest_map_obs()
