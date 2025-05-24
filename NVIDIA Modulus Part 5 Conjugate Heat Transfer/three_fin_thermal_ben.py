# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings

import torch
from sympy import Symbol, Eq, Abs, tanh, Or, And
import itertools
import numpy as np

import physicsnemo.sym
from physicsnemo.sym.hydra.config import PhysicsnemoConfig
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_3d import Box, Channel, Plane
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.basic import NormalDotVec, GradNormal
from physicsnemo.sym.eq.pdes.diffusion import Diffusion, DiffusionInterface
from physicsnemo.sym.eq.pdes.advection_diffusion import AdvectionDiffusion

from three_fin_geometry import *

import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter, InferencerPlotter
from mpl_toolkits.axes_grid1 import make_axes_locatable

class BaseValidatorPlotter(ValidatorPlotter):

    @staticmethod
    def heat_sink_mask_zy(zi, yi):
        """Returns a boolean mask where heat sink regions should be set to NaN for y-z plane"""
        mask = np.zeros_like(yi, dtype=bool)  

        heat_sink_y_start = -0.5  # First fin Y position
        heat_sink_z_start = -0.3  # First fin Z position
        fin_thickness = 0.1
        fin_length = 0.6
        gap = 0.15
        base_y_start = -0.5
        base_z_start = -0.3
        base_y_end = -0.3
        base_z_end = 0.3

        # Mask for the three fins
        for i in range(3):
            fin_z_start = heat_sink_z_start + i * (fin_thickness + gap)
            fin_z_end = fin_z_start + fin_thickness
            mask |= ((yi > heat_sink_y_start) & (yi < heat_sink_y_start + fin_length) & 
                     (zi > fin_z_start) & (zi < fin_z_end))
        # Mask for the base
        mask |= ((yi > base_y_start) & (yi < base_y_end) & (zi > base_z_start) & (zi < base_z_end))
        return mask  

    @staticmethod
    def heat_sink_mask_xy(xi, yi):
        """Returns a boolean mask where heat sink regions should be set to NaN for x-y plane"""
        mask = np.zeros_like(xi, dtype=bool)  

        heat_sink_x_start = -1  # First fin X position
        heat_sink_y_start = -0.5  # First fin Y position
        heat_sink_x_end = 0  # End fin X position
        heat_sink_y_end = 0.1  # End fin Y position

        # Mask
        mask |= ((xi > heat_sink_x_start) & (xi < heat_sink_x_end) & 
                 (yi > heat_sink_y_start) & (yi < heat_sink_y_end))
        return mask  

    @staticmethod
    def interpolate_output(x, y, values, extent, mask):
        """Interpolates irregular points onto a mesh"""
        xi, yi = np.meshgrid(
            np.linspace(extent[0], extent[1], 100),
            np.linspace(extent[2], extent[3], 100),
            indexing="ij",
        )
        
        interpolated_values = [
            scipy.interpolate.griddata((x, y), value, (xi, yi), method='linear') for value in values
        ]
        
        interpolated_values = [np.nan_to_num(val, nan=np.nan) for val in interpolated_values]
        
        for i in range(len(interpolated_values)):
            interpolated_values[i][mask] = np.nan

        return interpolated_values

    def plot(self, x, y, c_true, c_pred, extent, colorbar_limits, titles, mask_func, mask_inverted=False, ax=None):
        """Creates the plot for the given data"""
        xi, yi = np.meshgrid(
            np.linspace(extent[0], extent[1], 100),
            np.linspace(extent[2], extent[3], 100),
            indexing="ij",
        )
        mask = mask_func(xi, yi)
        if mask_inverted:
            mask = ~mask  # Invert the mask if needed
        c_true, c_pred = self.interpolate_output(x, y, [c_true, c_pred], extent, mask)

        # Compute difference
        c_diff = c_true - c_pred

        # Data for subplots
        data = [c_pred, c_true, c_diff]

        # Loop through subplots
        for i in range(3):
            if i < 2:  # Apply fixed color limits to predicted & true values
                im = ax[i].imshow(data[i].T, origin="lower", extent=extent, cmap="jet",
                                  vmin=colorbar_limits[0], vmax=colorbar_limits[1])
            else:  # No fixed limits for difference plot
                im = ax[i].imshow(data[i].T, origin="lower", extent=extent, cmap="jet")

            ax[i].set_title(titles[i])
            ax[i].set_xlabel("z" if mask_func == self.heat_sink_mask_zy else "x")
            ax[i].set_ylabel("y")

            # Create a colorbar with the same height as the plot
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))  # Set colorbar ticks to integers with fewer bins

class fluidCustomValidatorPlotter(BaseValidatorPlotter):

    def plot_zy(self, invar, true_outvar, pred_outvar, axes):
        "Custom plotting function for validator (showing 'c' in y-z plane)"

        # Filter data for x = -0.5
        #x_filter = np.isclose(invar["x"][:, 0], -0.5)
        x_values = invar["x"][:, 0]
        x_filter = (x_values > -0.51) & (x_values < -0.49)
        z, y = invar["z"][x_filter, 0], invar["y"][x_filter, 0]
        c_true = true_outvar["theta_f"][x_filter, 0] * 273.15
        c_pred = pred_outvar["theta_f"][x_filter, 0] * 273.15

        # Define fixed extent
        extent_zy = (-0.5, 0.5, -0.5, 0.5)

        # Define color bar limits
        colorbar_limits = (19.5, 44)  # Celsius temperature range

        # Titles for subplots
        titles = ["physicsnemo: T", "OpenFOAM: T", "Difference: T"]

        self.plot(z, y, c_true, c_pred, extent_zy, colorbar_limits, titles, self.heat_sink_mask_zy, mask_inverted=False, ax=axes)

    def plot_xy(self, invar, true_outvar, pred_outvar, axes):
        "Custom plotting function for validator (showing 'c' in x-y plane)"

        # Filter data for z = 0
        #z_filter = np.isclose(invar["z"][:, 0], 0.01)
        z_values = invar["z"][:, 0]
        z_filter = (z_values > -0.01) & (z_values < 0.01)
        x, y = invar["x"][z_filter, 0], invar["y"][z_filter, 0]
        c_true = true_outvar["theta_f"][z_filter, 0] * 273.15
        c_pred = pred_outvar["theta_f"][z_filter, 0] * 273.15

        # Define fixed extent
        extent_xy = (-2.5, 2.5, -0.5, 0.5)

        # Define color bar limits
        colorbar_limits = (22, 42)  # Celsius temperature range

        # Titles for subplots
        titles = ["physicsnemo: T", "OpenFOAM: T", "Difference: T"]

        self.plot(x, y, c_true, c_pred, extent_xy, colorbar_limits, titles, self.heat_sink_mask_xy, mask_inverted=False, ax=axes)

    def __call__(self, invar, true_outvar, pred_outvar):
        """Generate both y-z and x-y plane plots in a 2x3 grid"""
        f, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=100)
        plt.suptitle("Heat sink 3D: Temperature Comparison (physicsnemo vs OpenFOAM)")

        # Plot z-y plane
        self.plot_zy(invar, true_outvar, pred_outvar, axes[0, :])

        # Plot x-y plane
        self.plot_xy(invar, true_outvar, pred_outvar, axes[1, :])

        plt.tight_layout(pad=2.0)  # Adjust padding

        return [(f, "custom_plot")]

class solidCustomValidatorPlotter(BaseValidatorPlotter):

    def __call__(self, invar, true_outvar, pred_outvar):
        "Custom plotting function for validator (showing 'c' and heat sink regions)"

        # Filter data for x = -0.5
        mask = np.isclose(invar["x"][:, 0], -0.5)
        z, y = invar["z"][mask, 0], invar["y"][mask, 0]
        c_true = true_outvar["theta_s"][mask, 0] * 273.15
        c_pred = pred_outvar["theta_s"][mask, 0] * 273.15

        # Define fixed extent
        extent = (-0.5, 0.5, -0.5, 0.5)

        # Define color bar limits
        colorbar_limits = (20, 80)  # Celsius temperature range

        # Titles for subplots
        titles = ["physicsnemo: T", "OpenFOAM: T", "Difference: T"]

        f, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
        plt.suptitle("Heat sink 3D: Temperature Comparison (physicsnemo vs OpenFOAM)")

        self.plot(z, y, c_true, c_pred, extent, colorbar_limits, titles, self.heat_sink_mask_zy, mask_inverted=True, ax=axes)

        plt.tight_layout(pad=2.0)  # Adjust padding

        return [(f, "custom_plot")]

@physicsnemo.sym.main(config_path="conf", config_name="conf_thermal")
def run(cfg: physicsnemoConfig) -> None:
    # make thermal equations
    ad = AdvectionDiffusion(T="theta_f", rho=1.0, D=0.02, dim=3, time=False)
    dif = Diffusion(T="theta_s", D=0.0625, dim=3, time=False)
    dif_inteface = DiffusionInterface("theta_f", "theta_s", 1.0, 5.0, dim=3, time=False)
    f_grad = GradNormal("theta_f", dim=3, time=False)
    s_grad = GradNormal("theta_s", dim=3, time=False)

    # make network arch
    if cfg.custom.parameterized:
        input_keys = [
            Key("x"),
            Key("y"),
            Key("z"),
            Key("fin_height_m"),
            Key("fin_height_s"),
            Key("fin_length_m"),
            Key("fin_length_s"),
            Key("fin_thickness_m"),
            Key("fin_thickness_s"),
        ]
    else:
        input_keys = [Key("x"), Key("y"), Key("z")]
    flow_net = FullyConnectedArch(
        input_keys=input_keys,
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
    )
    thermal_f_net = FullyConnectedArch(
        input_keys=input_keys, output_keys=[Key("theta_f")]
    )
    thermal_s_net = FullyConnectedArch(
        input_keys=input_keys, output_keys=[Key("theta_s")]
    )

    # make list of nodes to unroll graph on
    thermal_nodes = (
        ad.make_nodes()
        + dif.make_nodes()
        + dif_inteface.make_nodes()
        + f_grad.make_nodes()
        + s_grad.make_nodes()
        + [flow_net.make_node(name="flow_network", optimize=False)]
        + [thermal_f_net.make_node(name="thermal_f_network")]
        + [thermal_s_net.make_node(name="thermal_s_network")]
    )

    geo = ThreeFin(parameterized=cfg.custom.parameterized)

    # params for simulation
    # heat params
    inlet_t = 293.15 / 273.15 - 1.0
    grad_t = 360 / 273.15

    # make flow domain
    thermal_domain = Domain()

    # inlet
    constraint_inlet = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=geo.inlet,
        outvar={"theta_f": inlet_t},
        batch_size=cfg.batch_size.Inlet,
        criteria=Eq(x, channel_origin[0]),
        lambda_weighting={"theta_f": 1.0},  # weight zero on edges
        parameterization=geo.pr,
    )
    thermal_domain.add_constraint(constraint_inlet, "inlet")

    # outlet
    constraint_outlet = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=geo.outlet,
        outvar={"normal_gradient_theta_f": 0},
        batch_size=cfg.batch_size.Outlet,
        criteria=Eq(x, channel_origin[0] + channel_dim[0]),
        lambda_weighting={"normal_gradient_theta_f": 1.0},  # weight zero on edges
        parameterization=geo.pr,
    )
    thermal_domain.add_constraint(constraint_outlet, "outlet")

    # channel walls insulating
    def wall_criteria(invar, params):
        sdf = geo.three_fin.sdf(invar, params)
        return np.less(sdf["sdf"], -1e-5)

    channel_walls = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=geo.channel,
        outvar={"normal_gradient_theta_f": 0},
        batch_size=cfg.batch_size.ChannelWalls,
        criteria=wall_criteria,
        lambda_weighting={"normal_gradient_theta_f": 1.0},
        parameterization=geo.pr,
    )
    thermal_domain.add_constraint(channel_walls, "channel_walls")

    # fluid solid interface
    def interface_criteria(invar, params):
        sdf = geo.channel.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    fluid_solid_interface = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=geo.three_fin,
        outvar={
            "diffusion_interface_dirichlet_theta_f_theta_s": 0,
            "diffusion_interface_neumann_theta_f_theta_s": 0,
        },
        batch_size=cfg.batch_size.SolidInterface,
        criteria=interface_criteria,
        parameterization=geo.pr,
    )
    thermal_domain.add_constraint(fluid_solid_interface, "fluid_solid_interface")

    # heat source
    sharpen_tanh = 60.0
    source_func_xl = (tanh(sharpen_tanh * (x - source_origin[0])) + 1.0) / 2.0
    source_func_xh = (
        tanh(sharpen_tanh * ((source_origin[0] + source_dim[0]) - x)) + 1.0
    ) / 2.0
    source_func_zl = (tanh(sharpen_tanh * (z - source_origin[2])) + 1.0) / 2.0
    source_func_zh = (
        tanh(sharpen_tanh * ((source_origin[2] + source_dim[2]) - z)) + 1.0
    ) / 2.0
    gradient_normal = (
        grad_t * source_func_xl * source_func_xh * source_func_zl * source_func_zh
    )
    heat_source = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=geo.three_fin,
        outvar={"normal_gradient_theta_s": gradient_normal},
        batch_size=cfg.batch_size.HeatSource,
        criteria=Eq(y, source_origin[1]),
    )
    thermal_domain.add_constraint(heat_source, "heat_source")

    # flow interior low res away from three fin
    lr_flow_interior = PointwiseInteriorConstraint(
        nodes=thermal_nodes,
        geometry=geo.geo,
        outvar={"advection_diffusion_theta_f": 0},
        batch_size=cfg.batch_size.InteriorLR,
        criteria=Or(x < -1.1, x > 0.5),
    )
    thermal_domain.add_constraint(lr_flow_interior, "lr_flow_interior")

    # flow interiror high res near three fin
    hr_flow_interior = PointwiseInteriorConstraint(
        nodes=thermal_nodes,
        geometry=geo.geo,
        outvar={"advection_diffusion_theta_f": 0},
        batch_size=cfg.batch_size.InteriorHR,
        criteria=And(x > -1.1, x < 0.5),
    )
    thermal_domain.add_constraint(hr_flow_interior, "hr_flow_interior")

    # solid interior
    solid_interior = PointwiseInteriorConstraint(
        nodes=thermal_nodes,
        geometry=geo.three_fin,
        outvar={"diffusion_theta_s": 0},
        batch_size=cfg.batch_size.SolidInterior,
        lambda_weighting={"diffusion_theta_s": 100.0},
    )
    thermal_domain.add_constraint(solid_interior, "solid_interior")

    # flow validation data
    file_path = "openfoam/"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {
            "Points:0": "x",
            "Points:1": "y",
            "Points:2": "z",
            "U:0": "u",
            "U:1": "v",
            "U:2": "w",
            "p_rgh": "p",
            "T": "theta_f",
        }
        if cfg.custom.turbulent:
            openfoam_var = csv_to_dict(
                to_absolute_path("openfoam/threeFin_extend_zeroEq_re500_fluid.csv"),
                mapping,
            )
        else:
            openfoam_var = csv_to_dict(
                to_absolute_path("openfoam/threeFin_extend_fluid0.csv"), mapping
            )
        openfoam_var["theta_f"] = (
            openfoam_var["theta_f"] / 273.15 - 1.0
        )  # normalize heat
        openfoam_var["x"] = openfoam_var["x"] + channel_origin[0]
        openfoam_var["y"] = openfoam_var["y"] + channel_origin[1]
        openfoam_var["z"] = openfoam_var["z"] + channel_origin[2]
        openfoam_var.update({"fin_height_m": np.full_like(openfoam_var["x"], 0.4)})
        openfoam_var.update({"fin_height_s": np.full_like(openfoam_var["x"], 0.4)})
        openfoam_var.update({"fin_thickness_m": np.full_like(openfoam_var["x"], 0.1)})
        openfoam_var.update({"fin_thickness_s": np.full_like(openfoam_var["x"], 0.1)})
        openfoam_var.update({"fin_length_m": np.full_like(openfoam_var["x"], 1.0)})
        openfoam_var.update({"fin_length_s": np.full_like(openfoam_var["x"], 1.0)})
        openfoam_invar_numpy = {
            key: value
            for key, value in openfoam_var.items()
            if key
            in [
                "x",
                "y",
                "z",
                "fin_height_m",
                "fin_height_s",
                "fin_thickness_m",
                "fin_thickness_s",
                "fin_length_m",
                "fin_length_s",
            ]
        }
        openfoam_flow_outvar_numpy = {
            key: value
            for key, value in openfoam_var.items()
            if key in ["u", "v", "w", "p"]
        }
        openfoam_thermal_outvar_numpy = {
            key: value
            for key, value in openfoam_var.items()
            if key in ["u", "v", "w", "p", "theta_f"]
        }
        openfoam_flow_validator = PointwiseValidator(
            nodes=thermal_nodes,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_thermal_outvar_numpy,
            plotter=fluidCustomValidatorPlotter(), # add
            batch_size=1024,# add
            requires_grad=True,# add
        )
        thermal_domain.add_validator(
            openfoam_flow_validator,
            "thermal_flow_data",
        )

        # solid data
        mapping = {"Points:0": "x", "Points:1": "y", "Points:2": "z", "T": "theta_s"}
        if cfg.custom.turbulent:
            openfoam_var = csv_to_dict(
                to_absolute_path("openfoam/threeFin_extend_zeroEq_re500_solid.csv"),
                mapping,
            )
        else:
            openfoam_var = csv_to_dict(
                to_absolute_path("openfoam/threeFin_extend_solid0.csv"), mapping
            )
        openfoam_var["theta_s"] = (
            openfoam_var["theta_s"] / 273.15 - 1.0
        )  # normalize heat
        openfoam_var["x"] = openfoam_var["x"] + channel_origin[0]
        openfoam_var["y"] = openfoam_var["y"] + channel_origin[1]
        openfoam_var["z"] = openfoam_var["z"] + channel_origin[2]
        openfoam_var.update({"fin_height_m": np.full_like(openfoam_var["x"], 0.4)})
        openfoam_var.update({"fin_height_s": np.full_like(openfoam_var["x"], 0.4)})
        openfoam_var.update({"fin_thickness_m": np.full_like(openfoam_var["x"], 0.1)})
        openfoam_var.update({"fin_thickness_s": np.full_like(openfoam_var["x"], 0.1)})
        openfoam_var.update({"fin_length_m": np.full_like(openfoam_var["x"], 1.0)})
        openfoam_var.update({"fin_length_s": np.full_like(openfoam_var["x"], 1.0)})
        openfoam_invar_solid_numpy = {
            key: value
            for key, value in openfoam_var.items()
            if key
            in [
                "x",
                "y",
                "z",
                "fin_height_m",
                "fin_height_s",
                "fin_thickness_m",
                "fin_thickness_s",
                "fin_length_m",
                "fin_length_s",
            ]
        }
        openfoam_outvar_solid_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["theta_s"]
        }
        openfoam_solid_validator = PointwiseValidator(
            nodes=thermal_nodes,
            invar=openfoam_invar_solid_numpy,
            #true_outvar=openfoam_thermal_outvar_numpy,
            true_outvar=openfoam_outvar_solid_numpy,# modify?
            plotter=solidCustomValidatorPlotter(), # add
            batch_size=1024,# add
            requires_grad=True,# add
        )
        thermal_domain.add_validator(
            openfoam_solid_validator,
            "thermal_solid_data",
        )
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_sym_examples_supplemental_materials"
        )
    # add peak temp monitors for design optimization
    # run only for parameterized cases and in eval mode
    if cfg.custom.parameterized and cfg.run_mode == "eval":
        # define candidate designs
        num_samples = cfg.custom.num_samples
        inference_param_tuple = itertools.product(
            np.linspace(*height_m_range, num_samples),
            np.linspace(*height_s_range, num_samples),
            np.linspace(*length_m_range, num_samples),
            np.linspace(*length_s_range, num_samples),
            np.linspace(*thickness_m_range, num_samples),
            np.linspace(*thickness_s_range, num_samples),
        )
        for (
            HS_height_m_,
            HS_height_s_,
            HS_length_m_,
            HS_length_s_,
            HS_thickness_m_,
            HS_thickness_s_,
        ) in inference_param_tuple:
            HS_height_m = float(HS_height_m_)
            HS_height_s = float(HS_height_s_)
            HS_length_m = float(HS_length_m_)
            HS_length_s = float(HS_length_s_)
            HS_thickness_m = float(HS_thickness_m_)
            HS_thickness_s = float(HS_thickness_s_)
            specific_param_ranges = {
                fin_height_m: HS_height_m,
                fin_height_s: HS_height_s,
                fin_length_m: HS_length_m,
                fin_length_s: HS_length_s,
                fin_thickness_m: HS_thickness_m,
                fin_thickness_s: HS_thickness_s,
            }

            # add metrics for peak temperature
            plane_param_ranges = {**specific_param_ranges}
            metric = (
                "peak_temp"
                + str(HS_height_m)
                + "_"
                + str(HS_height_s)
                + "_"
                + str(HS_length_m)
                + "_"
                + str(HS_length_s)
                + "_"
                + str(HS_thickness_m)
                + "_"
                + str(HS_thickness_s)
            )
            invar_temp = geo.three_fin.sample_boundary(
                5000,
                criteria=Eq(y, source_origin[1]),
                parameterization=plane_param_ranges,
            )
            peak_temp_monitor = PointwiseMonitor(
                invar_temp,
                output_names=["theta_s"],
                metrics={metric: lambda var: torch.max(var["theta_s"])},
                nodes=thermal_nodes,
            )
            thermal_domain.add_monitor(peak_temp_monitor)

    # make solver
    thermal_slv = Solver(cfg, thermal_domain)

    # start thermal solver
    thermal_slv.solve()


if __name__ == "__main__":
    run()
