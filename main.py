import argparse
import math
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    import quaternion

import pygeos
from shapely import geometry

import geopandas as gp
from geopandas import GeoSeries, GeoDataFrame
from geopandas.array import GeometryArray

from pyproj import CRS
from pyproj import Transformer
from pyproj.enums import TransformDirection

import cartopy


def main():
    parser = argparse.ArgumentParser('Plots shore lines using a perspective projection')
    parser.add_argument('--image', help="image on top of which the map is projected")
    parser.add_argument('--fov', type=float, help="fov in width")
    parser.add_argument('--width', type=int, help="image width in pixels")
    parser.add_argument('--height', type=int, help="image height in pixels")
    parser.add_argument('--lat', type=float, help="satellite latitude")
    parser.add_argument('--lon', type=float, help="satellite longitude")
    parser.add_argument('--alt', type=float, help="satellite altitude")
    parser.add_argument('--head', type=float, help="satellite heading: 0=north, 90=east, 180=south")
    parser.add_argument('--tilt', type=float, help="satellite tilt: 0=down, 90=flight direction")
    parser.add_argument('--roll', type=float, help="satellite roll: 0 = \"up\" towards north (?), 90 = \"up\" towards east (?)")
    args = parser.parse_args()

    # 4326 is same as cartopy.crs.PlateCarree()
    crs4326 = CRS.from_epsg(4326)
    bbox = (args.lon - 45, args.lat - 20, args.lon + 45, args.lat + 20) if False else None
    sc = '10m'
    types = OrderedDict()

    if 0:
        # from open street maps
        coast = gp.read_file('zip://./coastlines-split-4326.zip!coastlines-split-4326/lines.shp', bbox=bbox)
        coast.set_crs(crs4326)
    else:
        cst = cartopy.feature.NaturalEarthFeature(category='physical', name='coastline', scale=sc)
        coast = GeoDataFrame(geometry=list(cst.geometries()), crs=crs4326)
    coast['type'] = ['coast']*len(coast)
    types['coast'] = dict(facecolor='none', edgecolor='black', linewidth=1, alpha=0.5)

    if 0:
        # from https://gadm.org/download_country_v3.html
        borders = gp.read_file('zip://./gadm36_FIN_shp.zip!gadm36_FIN_0.shp', bbox=bbox)
        borders.set_crs(crs4326)
    else:
        brds = cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=sc)
        borders = GeoDataFrame(geometry=list(brds.geometries()), crs=crs4326)
    borders['type'] = ['border']*len(borders)
    types['border'] = dict(facecolor='none', edgecolor='black', linewidth=1, alpha=0.5)

    # TODO: add horizon
    # TODO: add cities

    feats = coast.append(borders, ignore_index=True)

    if 0:
        # magnetic declination (9.67 deg east) using this: https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml
        slice = get_meas_slice((78.148, 16.043, 520), 9.67, (150, 175), crs4326, dist=3e6)  # (150, 170)
        slice['type'] = ['slice'] * len(slice)
        types['slice'] = dict(facecolor='none', edgecolor='white', linestyle='--', linewidth=1, alpha=0.5)
        feats = feats.append(slice, ignore_index=True)

    perspective_projection(feats,
                           (args.lat, args.lon, args.alt),
                           (args.head, args.tilt, args.roll+180),
                           args.fov, args.width, args.height)

    if args.image:
        img = plt.imread(args.image)
        ax = plt.imshow(img).axes
    else:
        fig, ax = plt.subplots()

    for type, styling in types.items():
        feats.loc[feats['type'] == type].plot(ax=ax, **styling)
    ax.set_xlim(0, args.width)
    ax.set_ylim(args.height, 0)
    plt.show()
    print('finished')


def get_meas_slice(sensor_coords, mag_decl, angles, crs, dist=3e6):
    lat, lon, alt = sensor_coords
    crs_cart = CRS.from_string('+proj=cart')
    transformer = Transformer.from_crs(crs, crs_cart)
    sensor_loc = np.array(transformer.transform(lat, lon, alt))
    xu_zn_q = ypr_to_q(lat, lon, 0)

    def get_points(angle, n=10):
        u = q_times_v(xu_zn_q * eul_to_q((-mag_decl, angle - 90), 'xy'), np.array([1, 0, 0]))
        return [sensor_loc + u * d for d in np.linspace(0, dist, n)]

    def coords(points):
        arr = []
        for pt in points:
            lat, lon, alt = transformer.transform(*pt, direction=TransformDirection.INVERSE)
            arr.append([lon, lat, alt])
        return arr

    lines = [geometry.LineString(coordinates=coords(get_points(angle))) for angle in angles]
    df = GeoDataFrame(geometry=lines, crs=crs)
    return df


def swap_lat_lon(df):
    data = df.geometry.values.data
    coords = pygeos.get_coordinates(data)
    pygeos.set_coordinates(data, coords[:, (1, 0)])


def perspective_projection(df, sc_coords, sc_heading, fov, width, height, debug=False):
    lat, lon, alt = sc_coords
    head, tilt, roll = sc_heading

    crs_cart = CRS.from_string('+proj=cart')
    transformer = Transformer.from_crs(df.crs, crs_cart)

    # cartesian coordinates:
    #   x-axis points from the Earth center to the point of longitude=0, latitude=0
    #   y-axis points from the Earth center to the point of longitude=90, latitude=0
    #   z-axis points to the North pole

    # satellite location in cartesian coords
    sc_pos_v = np.array(transformer.transform(lat, lon, alt))

    # get rotation and projection matrices
    rot_mx = get_rot_mx(lat, lon, head, tilt, roll)
    cam_mx = get_cam_mx(fov, width, height)

    if debug:
        # test with svalbard coords
        sval = np.array(transformer.transform(78.148, 16.043, 520))
        sc_sv = sval - sc_pos_v
        sc_sv_cf = rot_mx.dot(sc_sv) * 1e-3
        sv_im = cam_mx.dot(sc_sv_cf)
        print('svalbard at [km] %s, in image [px]: %s' % (sc_sv_cf, sv_im[:2]/sv_im[2]))

    data = df.geometry.values.data
    coords = pygeos.get_coordinates(data, include_z=True)   # order out: lon, lat
    shape = coords.shape
    fc = coords.flatten()
    fc[np.isnan(fc)] = 0
    coords = fc.reshape(shape)
    loc = transformer.transform(coords[:, 1], coords[:, 0], coords[:, 2])  # order in: lat, lon
    loc = np.stack(loc, axis=1)
    rel_loc = loc - sc_pos_v

    # remove features that are farther away than the horizon
    r = np.linalg.norm(loc[0, :])
    horizon_dist = math.sqrt((r + alt)**2 - r**2)
    mask = np.linalg.norm(rel_loc, axis=1) > horizon_dist
    rel_loc[mask, :] = np.nan

    # rotate and project to image
    img_coords = cam_mx.dot(rot_mx).dot(rel_loc.T).T
    img_coords = img_coords[:, :2] / img_coords[:, 2:]
    new_data = pygeos.set_coordinates(data.copy(), img_coords)
    new_geom = GeoSeries(GeometryArray(new_data), crs=crs_cart, name=df.geometry.name)
    df.geometry = new_geom[new_geom.is_valid]


def get_rot_mx(lat, lon, head, tilt, roll):
    # first point +x-axis to satellite location, +z-axis towards north
    xu_zn_q = ypr_to_q(lat, lon, 0)

    # then point -z-axis to look at target direction, +y-axis is up
    rel_rot_q = xu_zn_q * eul_to_q((180 - head, 90 - tilt, roll - 90), 'xyz')

    # get rotation matrix for affine_transform
    return quaternion.as_rotation_matrix(rel_rot_q.conj())


def get_cam_mx(fov, width, height):
    # cam mx for opengl -z axis looking camera
    fov_h, fov_v = fov, fov * height / width
    x = width / 2
    y = height / 2
    fl_x = x / math.tan(math.radians(fov_h) / 2)
    fl_y = y / math.tan(math.radians(fov_v) / 2)
    cam_mx = np.array([[-fl_x, 0, x],
                       [0,  fl_y, y],
                       [0,     0, 1]])
    return cam_mx


def ypr_to_q(lat, lon, roll, deg=True):
    if deg:
        lat = math.radians(lat)
        lon = math.radians(lon)
        roll = math.radians(roll)

    # Tait-Bryan angles, aka yaw-pitch-roll, nautical angles, cardan angles
    # intrinsic euler rotations z-y'-x'', pitch=-lat, yaw=lon
    return (
            np.quaternion(math.cos(lon / 2), 0, 0, math.sin(lon / 2))
            * np.quaternion(math.cos(-lat / 2), 0, math.sin(-lat / 2), 0)
            * np.quaternion(math.cos(roll / 2), math.sin(roll / 2), 0, 0)
    )


def eul_to_q(angles, order='xyz', reverse=False, deg=True):
    assert len(angles) == len(order), 'len(angles) != len(order)'

    if deg:
        angles = list(map(math.radians, angles))

    q = quaternion.one
    idx = {'x': 0, 'y': 1, 'z': 2}
    for angle, axis in zip(angles, order):
        w = math.cos(angle / 2)
        v = [0, 0, 0]
        v[idx[axis]] = math.sin(angle / 2)
        dq = np.quaternion(w, *v)
        q = (dq * q) if reverse else (q * dq)
    return q


def q_times_v(q, v):
    qv = np.quaternion(0, *v)
    qv2 = q * qv * q.conj()
    return np.array([qv2.x, qv2.y, qv2.z])


if __name__ == '__main__':
    main()

