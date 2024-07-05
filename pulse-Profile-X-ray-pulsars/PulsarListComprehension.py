# Here I will combine parts of the other lighthouse files to fit simulated beam patterns to the observed beam pattern.
import time

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

"""This python file describes all the functions regarding the physics of the x-ray pulsars emission.
Furthermore it calculates a simulated hotspot pattern for each parameters.

At the end the simulated data are stored in a list of dictionaries, where all the simulated parameters are stored, 
alongside with the correspondent pulse profile.

These functions are the basic data creation for the main() file. The data could be saved in a parquet file decommenting the last lines"""


def cart2sph(x, y, z):
    xy = np.sqrt(x ** 2 + y ** 2)

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arctan2(y, x)
    theta = np.arctan2(xy, z)

    return r, theta, phi


def sph2cart(r, theta, phi):
    X = r * np.sin(theta) * np.cos(phi)
    Y = r * np.sin(theta) * np.sin(phi)
    Z = r * np.cos(theta)
    return X, Y, Z


def hotspotXYZ(RotationInclination, RotationAzimut, MagnColatitude, npoints, shift=0):

    """This function defines the position of the hotspot.

        Parameters:
            ----------
            RotationInclination: how much is the rotation axis inclined in deg

            RotationAzimuth: the azimuthal angle that defines the rotation axis in deg

            MagnColatitude: Angular position of the hotspot with respect to the rotation axis point, in deg.
            E.g. if between the rotational vector and the vector from the center of the sphere and the hotspot there is an angle of 30°,
            this will be the MagneticColatitude

            npoints: The number of points where the hotspot can lay around the rotational axis, it will be the same as the dimension as the decomposition values (32)

            shift: The phase shift due to a different angle of observation of the system, will give the same pattern but shifted, in rad

        Returns:
            ----------------
            hotspot:
                position of the points where the hotspot can be around the rotation axis (list of arrays)

            """
    from numpy.linalg import norm
    from scipy.spatial.transform import Rotation
    ##  We first locate the rotation axis creating the point on the sphere defined by the given angles  ##

    altitude = np.deg2rad(90 - RotationInclination)  # inclination 0-90 deg, from the horizon up
    azimuth = np.deg2rad(RotationAzimut)  # 0-2*pi, around a circle on the horizon
    rotax = sph2cart(1, altitude, azimuth)  # in cartesian coordinates
    rotax = rotax / norm(rotax)  # normalized

    ## --- Location of the hotspot --- ###
    HotspotAltitude = altitude + np.deg2rad(
        MagnColatitude)  # inclination 0-90 deg, from the horizon up, defined with respect to the altitude of the rotational angle
    HotspotAzimuth = azimuth  # 0-2*pi, around a circle on the horizon

    hotspot = sph2cart(1, HotspotAltitude, HotspotAzimuth)  # Locating the hotspot in cartesian coordinates
    hotspot = hotspot / norm(hotspot)  # Normalization of the vector to fit in the sphere of radius 1

    ## --- Rotation of the hotspot around the rotation axis --- ###
    phirot = np.linspace(0 + shift, 2 * np.pi + shift,
                         npoints).tolist()  # We create an array of equispaced angles based on the defined shift
    rot = [Rotation.from_rotvec(x * rotax) for x in
           phirot]  # Rotation.from_rotvec defines the parameters for a rotation around a point

    hotspot = [x.apply(hotspot) for x in
               rot]  # Creates different possible hotspots around the rotation axis, applying the rotation with x.apply(),
    # using hotspot we have the rotation based on the MagnColatitude, all the vectors defining this rotation will have a MagnColatitude angle with respect to the rotation axis
    return hotspot


def GravitationalLightBending(psi):
    """ Defines all the parameters needed for light bending using general relativity, and then returns the angle the observer sees

    Parameters:
        ------------
    psi: angle observed due to light bending (float)

    Returns:
        ------------
        alpha: actual angle of emission (float)

    Notes:
        See Beloborodov 2002 Equation 1 for futher information

        """
    G = 6.67e-8  # cm3 g−1 s−2
    M = 1.98847e33  # g
    mNS = 1.4  # in solar masses (M)
    lsc = 2.99792458e10  # cm/s
    rg = 2 * G * mNS * M / lsc ** 2 * 1e-5  # km
    r = 3 * rg  # 3*rg = 12; 10 # km   # 2/0.19 rg = 43.5 km

    alpha = np.arccos(
        1 - (1 - np.cos(psi)) * (1 - rg / r))  # Beloborodov 2002 Equation 1; psi in B02 = beta for me; alpha = alpha

    return alpha


def intrinsicAngles(hotspotLoc):
    """Transforms the observed angles, defined with the function hotspotXYZ, into
     the intrinsic angles, taking into account the gravitational light bending


     Parameters:
        -----------
        hotspotLoc: Location of the hotspot, list of arrays defining the angles of the positions of the hotspot around the rotation axis (list of arrays)

     Returns:
        ------------
        hotspotTheta: list of the possible observed azimuthal angles of the hotspot (list)

        hotspotPhi: list of the possible observed inclination angles of the hotspot (list)

        intrinsicTheta: possible intrinsic azimuthal angles of the hotspot, aster taking into account gravitational light bending (list)

        intrinsicPhi: possible intrinsic inclination angles of the hotspot, aster taking into account gravitational light bending (list)

     """
    hotspotTheta = [np.pi / 2 - cart2sph(*loc)[1] for loc in
                    hotspotLoc]  # Get the azimuthal angles from the location of the hotspots
    hotspotPhi = [cart2sph(*loc)[2] for loc in
                  hotspotLoc]  # Get the inclination angles from the location of the hotspots

    intrinsicTheta = GravitationalLightBending(hotspotTheta)  # Transforming theta with the gravitational bending
    intrinsicTheta = np.nan_to_num(intrinsicTheta, nan=np.pi)
    intrinsicTheta = np.where(np.array(hotspotTheta) < 0, -intrinsicTheta,
                              intrinsicTheta)  # need this because of geometrical symmetries that are not part of the equation in GravitationalLightBending

    intrinsicPhi = GravitationalLightBending(hotspotPhi)  # Transform phi with the gravitational light bending
    intrinsicPhi = np.nan_to_num(intrinsicPhi, nan=np.pi)
    intrinsicPhi = np.where(np.array(hotspotPhi) < 0, -intrinsicPhi,
                            intrinsicPhi)  # need this because of geometrical symmetries that are not part of the equation in GravitationalLightBending

    return hotspotTheta, hotspotPhi, intrinsicTheta, intrinsicPhi


def defBeampattern(irange, arange, grid1, grid2, param1, param2, param3, param4, param5):
    """Uses the parametrization of the beam pattern in order to place it on the grid defined by grid1 and grid2,

     irange and arange are the range of the angles that are used to define the grids that we mesh together afterward to create the whole grid to plot the beampattern on.

     This function also includes the asymmetry of the inclination and the azimuthal angles of the beam emission.

     these angles are defined with respect to the normal to the surface of the hotspot.

     Parameters:
        ---

        irange (list): range of the possible inclination angles (from 0 to pi/2), with respect to the normal of the hotspot

        arange (list): range of the possible azimuthal angles (from -pi/2 to pi/2), with respect to the y-axis (list)

        grid1 (list): X result from the meshgrid of irange and arange

        grid2 (list): Y result from the meshgrid of irange and arange

        param1 (float): define the shifting of the cosine functions in the equation of beampattern.

        param2 (float): define the shifting of the cosine functions in the equation of beampattern.

        param3 (float): define the exponent of the equation beampattern that allows to adjust the width or “peakiness” of the beam pattern

        param4 (float): Used to multiply the cosines of the irange and arange angles to introduce more asymmetry

        param5 (float): Used to multiply the cosines of the irange and arange angles to introduce more asymmetry



    Returns:

        ----------------

        beampattern: the pattern of the beam that is emitted from the pole of the pulsar, containing also the asymmetries (2D numpy array)



    References:

        -----

        fig 7.12, pag. 134 of Saathof, 2023

     """

    beampattern = (np.cos(grid2 + param1) * np.cos(grid1 + param2)) ** int(param3)
    # irange_asym = np.array(np.cos(irange) * param4)
    # arange_asym = np.array(np.cos(arange) * param5)
    # beampattern = beampattern.T
    # beampattern += irange_asym  # adding the asymmetries to the pattern calculated before
    # beampattern = beampattern.T
    # beampattern += arange_asym

    beampattern = np.abs(beampattern)
    beampattern /= np.max(beampattern)
    beampattern = np.where(grid1 > np.pi / 2, 0, beampattern)
    beampattern = np.where(grid2 > np.pi / 2, 0, beampattern)
    beampattern = np.where(grid1 < -np.pi / 2, 0, beampattern)
    beampattern = np.where(grid2 < -np.pi / 2, 0, beampattern)
    return beampattern


def f_parametrized(RotationInclination, RotationAzimuth, MagnColatitude, param1, param2, param3, param4, param5, shift):
    """Combines all the functions defined before:

    Parameters:
        -----------
        RotationInclination: angle of inclination of the rotation axis

        RotationAzimuth: azimuth angle of the rotation axis

        MagnColatitude: relative angle between the hotspot vector and the rotation axis

        param1, param2, param3, param4, param5: see the function "defBeamPattern"

        shift: phase shift of the pulse profile

    Returns:
        ---------------
        irange: range of the inclination angle of the beam emission

        arange: range of the azimuth angle of the beam emission

        grid1, grid2: grid defined by irange and arange, from meshgrid

        hotspotLoc: location of the hotspot, calculated with the function "hotspotXYZ"

        decomposition: the decomposition of the pulse profile defined by the function "decompositions"

        hotspotTheta: list of the possible observed azimuthal angles of the hotspot (list)

        hotspotPhi: list of the possible observed inclination angles of the hotspot (list)

        intrinsicTheta: possible intrinsic azimuthal angles of the hotspot, aster taking into account gravitational light bending (list)

        intrinsicPhi: possible intrinsic inclination angles of the hotspot, aster taking into account gravitational light bending (list)

        pattern: the pattern defined by the equations of the beam in "defBeamPattern" (2D numpy array)

        hotspotPattern: pattern obtained interpolating the function of the beamPattern with the angles that define the hotspot (list)
        """

    irange = np.linspace(-np.pi / 2, np.pi / 2, 32)
    arange = np.linspace(-np.pi / 2, np.pi / 2, 32)
    grid1, grid2 = np.meshgrid(irange, arange)
    npoints = 32

    hotspotLoc = hotspotXYZ(RotationInclination, RotationAzimuth, MagnColatitude, npoints, shift)

    hotspotTheta, hotspotPhi, intrinsicTheta, intrinsicPhi = intrinsicAngles(hotspotLoc)

    pattern = defBeampattern(irange, arange, grid1, grid2, param1, param2, param3, param4, param5)
    r = RectBivariateSpline(irange, arange, pattern.T)

    patternSpline = lambda x, y: r(x,
                                   y).T  # gives a function that can interpolate the grid (x y axis) with the beam pattern (z axis)

    hotspotPattern = [patternSpline(intrinsicTheta[i], intrinsicPhi[i]) for i in
                      range(len(intrinsicPhi))]  # Evaluate the interpolating function at the angles of the hotspot
    hotspotPattern = np.concatenate(hotspotPattern).ravel().tolist()

    return irange, arange, grid1, grid2, hotspotLoc, \
        hotspotTheta, hotspotPhi, intrinsicTheta, intrinsicPhi, \
        pattern, hotspotPattern, param1, param2, param3, param4, param5


# Define parameters
def ranges(i):
    """This function defines the ranges of variation of the parameters in order to create the data



    Parameters:

        ----------

        i (float): the number of iterations for each parameter



    Returns:

        ----------

        A list of len = i for each of the 7 parameters

        """

    rot_i = np.linspace(-90, 90, i + 3)  # More important, I added 3 iterations for now, there can be more

    rot_a = np.linspace(-150, 180, i,
                        endpoint=False)  # Not super interested if the hotspot is behind the star, exclude +-180

    magnc = np.linspace(0, 90, i)

    shift = np.linspace(-np.pi, np.pi, i,
                        endpoint=False)  # Set some endpoint= False because I think they would lead to the same result

    p1 = np.linspace(-np.pi, np.pi, i, endpoint=False)

    p2 = np.linspace(-np.pi, np.pi, i, endpoint=False)

    p3 = np.linspace(0, 20, 4)  # Less important, let us limit to 4 interactions

    # init_p4 = np.linspace(0, 1, 5) #These are asymmetry parameters, ignore for now

    # p5 = np.linspace(0, 1, 5)

    return rot_i, rot_a, magnc, shift, p1, p2, p3


def dataGenerator(number_data, file_path):
    """This function creates the dataset that is the goal of the project. It ranges over
    all the possible combinations of parameters and then creates the physical pulse profile with
    the function f_parametrized. Then it saves a list of dictionaries containing all the data

    Parameters:
        ----------
        number_data: the number of iterations for each parameter that will be performed
           by the function ranges. (float)

    Returns:
        ----------
        The list of dictionaries containing all the simulated data, and it also prints the time it takes in minutes"""
    #   Determine the ranges of variation of the parameters
    rot_i, rot_a, magnc, shift, p1, p2, p3 = ranges(number_data)

    # Iterating over all the possible combinations of parameters
    start = time.time()
    # The following lines evaluate the function with the different set of parameters decided
    # Use a list comprehension and a progress bar to check the results

    with tqdm(total=len(p1) * len(p2) * len(p3) * len(rot_i) * len(rot_a) * len(magnc) * len(shift)) as pbar:
        data = [
            {
                'param1': param1,
                'param2': param2,
                'param3': param3,
                'hotspot pattern': hotspotPattern,
                'rotation inclination': rot_i[q],
                'rotation azimuth': rot_a[t],
                'magnetic colatitude': magnc[u],
                'shift': shift[v]
            }
            for i in range(len(p1))
            for j in range(len(p2))
            for k in range(len(p3))
            for q in range(len(rot_i))
            for t in range(len(rot_a))
            for u in range(len(magnc))
            for v in range(len(shift))
            for (irange, arange, grid1, grid2, hotspotLoc, hotspotTheta, hotspotPhi, intrinsicTheta, intrinsicPhi,
                 beampattern, hotspotPattern, param1, param2, param3, param4, param5)
            in [f_parametrized(rot_i[q], rot_a[t], magnc[u], p1[i], p2[j], p3[k], 0, 0, shift[v])]
            if not pbar.update(1)  # Update the progress bar
        ]

    end = time.time()
    print("Time in minutes: ", (end - start) / 60)

    # Create the dataframe and save it to a file
    df = pd.DataFrame(data)
    df.to_parquet(file_path, engine='pyarrow', index=False)
    return data

### Decomment to make this script creating a file ###


# i = 5  # Choose the length of the ranges of the parameters
# file_path = ""
# data = dataGenerator(i, file_path)

# Check that data are indeed saved in the list
# print("Size: ", np.size(data))
