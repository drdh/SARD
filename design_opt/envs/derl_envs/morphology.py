import os
import random
from collections import defaultdict

import imageio
import numpy as np
from lxml import etree

from design_opt.envs.derl_envs.config import cfg
from design_opt.envs.derl_envs.utils import file as fu
from design_opt.envs.derl_envs.utils import geom as gu
from design_opt.envs.derl_envs.utils import mjpy as mu
from design_opt.envs.derl_envs.utils import sample as su
from design_opt.envs.derl_envs.utils import xml as xu

HEAD = "torso/0"

class Morphology:
    """Representation for symmetric unimal."""

    def __init__(self, id_, robot):
        self.id = id_
        self.robot = robot
        self.cur_body = self.robot.bodies[0]
        self.body_name2site_name = {}
        self._init_new_unimal()
        self.parent_id = ""

        self.worldbody.append(self.unimal)
        for self.cur_body in self.robot.bodies[1:]:
            self.grow()

        self.sim = mu.mjsim_from_etree(self.root)
        self.sim.step()
        new_contacts = self._new_contacts(self.sim, self.exclude_geom_pairs_list)
        # Update contacts
        new_contacts = self._contact_id2name(self.sim, new_contacts)
        self.contact_pairs = self.contact_pairs.union(new_contacts)

        self.set_head_pos()
        self._before_save()


    def _init_new_unimal(self):
        """Construct a new unimal with just head."""
        self.init_state = None
        # Initalize xml from template
        self.root, self.tree = xu.etree_from_xml(cfg.UNIMAL_TEMPLATE)
        self.worldbody = self.root.findall("./worldbody")[0]
        self.actuator = self.root.findall("./actuator")[0]
        self.contact = self.root.findall("./contact")[0]

        # Set of contact pairs. Each item is a list (geom_name, geom_name).
        # Note: We cannot store ids as there is not gurantee they will be the
        # same.
        self.contact_pairs = set()
        self.exclude_geom_pairs_list = []

        self.num_limbs = 0
        # In case of delete_limb op limb_idx can differ from num_limbs
        self.limb_idx = 0
        # Unimals starts with a head.
        self.num_torso = 1

        # List of geom names. e.g [[limb1], [limb2, limb3], [limb4]] where
        # limb2 and limb3 are symmetric counterparts.
        self.limb_list = []

        # List of torso names
        self.torso_list = [0]

        # List of torso from where limbs can grow
        self.growth_torso = [0]

        # Body params
        self.body_params = {
            "torso_mode": "vertical",
            "torso_density": 500,
            "limb_density": 500,
            "num_torso": 0,
        }

        # Contains information about a limb like: orientation, parent etc
        self.limb_metadata = defaultdict(dict)

        # Construct unimal
        self.unimal = self._construct_head()


    def _construct_head(self):
        # Placeholder position which will be updated based final unimal height
        head = xu.body_elem("torso/0", [0, 0, 0])
        # Add joints between unimal and the env (generally the floor)
        if cfg.HFIELD.DIM == 1:
            head.append(xu.joint_elem("rootx", "slide", "free", axis=[1, 0, 0]))
            head.append(xu.joint_elem("rootz", "slide", "free", axis=[0, 0, 1]))
            head.append(xu.joint_elem("rooty", "hinge", "free", axis=[0, 1, 0]))
        else:
            head.append(xu.joint_elem("root", "free", "free"))

        head.append(xu.site_elem("root", None, "imu_vel"))

        head_params = self._choose_torso_params()
        r = head_params["torso_radius"]
        # Add the actual head
        head.append(
            etree.Element(
                "geom",
                {
                    "name": HEAD,
                    "type": "sphere",
                    "size": "{}".format(r),
                    "condim": str(cfg.TORSO.CONDIM),
                    "density": str(head_params["torso_density"]),
                },
            )
        )
        # Add cameras
        head.append(
            etree.fromstring(
                '<camera name="side" pos="0 -7 2" xyaxes="1 0 0 0 1 2" mode="trackcom"/>'
            )
        )

        # Add site where limb can be attached
        size_name = "torso/0"
        head.append(xu.site_elem(size_name, [0, 0, 0], "growth_site"))
        self.body_name2site_name[self.cur_body.name] = size_name
        # Add btm pos site
        head.append(xu.site_elem("torso/btm_pos/0", [0, 0, -r], "btm_pos_site"))
        # Add site for touch sensor
        head.append(
            xu.site_elem("torso/touch/0", None, "touch_site", str(r + 0.01))
        )
        head.append(
            xu.site_elem("torso/vertical/0", [0, 0, -r], "torso_growth_site",) # grow new torso?
        )
        return head

    def _choose_torso_params(self):
        return {
            "torso_radius": float(self.cur_body.geoms[0].node.attrib['size']), # su.sample_from_range(cfg.TORSO.HEAD_RADIUS_RANGE),
            "torso_density": self.body_params["torso_density"],
        }

    def _construct_limb(self, idx, site, site_type, limb_params):
        # Get parent radius
        parent_idx = xu.name2id(site)
        if "torso" in site.get("name"):
            parent_name = "torso/{}".format(parent_idx)
        else:
            parent_name = "limb/{}".format(parent_idx)
        p_r = self._get_limb_r(parent_name)

        # Get limb_params
        r, h = limb_params["limb_radius"], limb_params["limb_height"]
        h, theta, phi = orient = limb_params["limb_orient"]

        # Set limb pos
        pos = xu.str2arr(site.get("pos"))
        pos = xu.add_list(pos, gu.sph2cart(p_r, theta, phi))

        name = "limb/{}".format(idx)
        limb = xu.body_elem(name, pos)

        theta_j = np.pi + theta
        if theta_j >= 2 * np.pi:
            theta_j = theta_j - 2 * np.pi
        joint_pos = gu.sph2cart(r, theta_j, np.pi - phi)
        limb.append(
            xu.joint_elem(
                "limb{}/{}".format(limb_params["joint_axis"], idx),
                "hinge",
                "normal_joint",
                axis=limb_params["joint_axis3"], #xu.axis2arr(axis),
                range_=xu.arr2str(limb_params["joint_range"][0]),
                pos=xu.arr2str(joint_pos),
            )
        )

        # Create from, to points
        x_f, y_f, z_f = [0.0, 0.0, 0.0]
        x_t, y_t, z_t = gu.sph2cart(r + h, theta, phi)

        # Create actual geom
        # Note as per mujoco docs: The elongated part of the geom connects the
        # two from/to points i.e so we have to handle the r (and p_r) for
        # all the positions.
        limb.append(
            etree.Element(
                "geom",
                {
                    "name": name,
                    "type": "capsule",
                    "fromto": xu.arr2str([x_f, y_f, z_f, x_t, y_t, z_t]),
                    "size": "{}".format(limb_params["limb_radius"]),
                    "density": str(limb_params["limb_density"]),
                },
            )
        )
        x_mid, y_mid, z_mid = gu.sph2cart(r + h / 2, theta, phi)
        limb.append(
            xu.site_elem(
                "limb/mid/{}".format(idx), [x_mid, y_mid, z_mid], site_type
            )
        )

        x_end, y_end, z_end = gu.sph2cart(r + h, theta, phi)
        site_name = "limb/btm/{}".format(idx)
        limb.append(
            xu.site_elem(
                site_name, [x_end, y_end, z_end], site_type
            )
        )
        self.body_name2site_name[self.cur_body.name] = site_name
        # Site to determine bottom position of geom in global coordinates
        x_end, y_end, z_end = gu.sph2cart(2 * r + h, theta, phi)
        limb.append(
            xu.site_elem(
                "limb/btm_pos/{}".format(idx),
                [x_end, y_end, z_end],
                "btm_pos_site",
            )
        )

        # Site for touch sensor
        limb.append(
            xu.site_elem(
                "limb/touch/{}".format(idx),
                None,
                "touch_site",
                "{}".format(limb_params["limb_radius"] + 0.01),
                xu.arr2str([x_f, y_f, z_f, x_t, y_t, z_t]),
                "capsule",
            )
        )
        return limb, orient

    def set_head_pos(self):
        # Set the head pos to 0, 0, 0 before loading mjsim
        self.unimal.set("pos", xu.arr2str([0.0, 0.0, 0.0]))

        # sim = mu.mjsim_from_etree(self.root)
        sim = self.sim
        btm_pos_sites = xu.find_elem(self.unimal, "site", "class", "btm_pos_site")
        btm_pos_site_ids = [
            mu.mj_name2id(sim, "site", site.get("name")) for site in btm_pos_sites
        ]
        # sim.step()

        # Get z_axis of all the sites
        z_coords = sim.data.site_xpos[btm_pos_site_ids][:, 2]
        # All the z_axis are < 0. Select the smallest
        btm_most_pos_idx = np.argmin(z_coords)
        head_z = -1 * z_coords[btm_most_pos_idx] + cfg.BODY.FLOOR_OFFSET
        self.unimal.set("pos", xu.arr2str([0, 0, round(head_z, 2)]))

    def _add_actuator(self, body_type, idx, params):
        # for axis, gear in zip(params["joint_axis"], params["gear"]):
        name = "{}{}/{}".format(body_type, params["joint_axis"], idx)
        self.actuator.append(xu.actuator_elem(name, params["gear"]))

    def _attach(self, site, body_part):
        parent = self.unimal.find(
            './/site[@name="{}"]...'.format(site.get("name"))
        )
        parent.append(body_part)
        return parent

    def _contact_name2id(self, sim, contacts):
        """Converts list of [(name1, name2), ...] to [(id1, id2), ...]."""
        return [
            (mu.mj_name2id(sim, "geom", name1), mu.mj_name2id(sim, "geom", name2))
            for (name1, name2) in contacts
        ]

    def _contact_id2name(self, sim, contacts):
        """Converts list of [(id1, id2), ...] to [(name1, name2), ...]."""
        return [
            (mu.mj_id2name(sim, "geom", id1), mu.mj_id2name(sim, "geom", id2))
            for (id1, id2) in contacts
        ]

    def to_string(self):
        return etree.tostring(self.root, encoding="unicode", pretty_print=True)

    def grow(self):
        # Find a site to grow
        choosen_site = self._choose_site()

        choosen_site, limbs2add = choosen_site
        site_type = choosen_site.get("class")

        # Choose limb params
        limb_params = self._choose_limb_params()

        # Get new limbs and sites where to attach them
        limbs, attach_sites, orients = self._get_new_limbs(
            choosen_site, limbs2add, limb_params
        )

        # Attach the limbs on the site
        parents = []
        exclude_geom_pairs = []

        for cur_limb, cur_site in zip(limbs, attach_sites):
            parents.append(self._attach(cur_site, cur_limb))
            exclude_geom_pairs.append((parents[-1], cur_limb))

        self.exclude_geom_pairs_list.extend(exclude_geom_pairs)

        # Add actuators
        for idx in range(self.limb_idx, self.limb_idx + limbs2add):
            self._add_actuator("limb", idx, limb_params)
        # Update limbs
        self.limb_list.append([xu.name2id(limb_) for limb_ in limbs])
        # Update metadata
        for (l, o, p, a) in zip(limbs, orients, parents, attach_sites):
            lp = limb_params
            limb_idx = xu.name2id(l)
            self.limb_metadata[limb_idx]["joint_axis"] = lp["joint_axis"]
            self.limb_metadata[limb_idx]["orient"] = o
            self.limb_metadata[limb_idx]["parent_name"] = p.get("name")
            self.limb_metadata[limb_idx]["site"] = a.get("name")
            self.limb_metadata[limb_idx]["gear"] = {}
            for axis, gear in zip(lp["joint_axis"], lp["gear"]):
                self.limb_metadata[limb_idx]["gear"][axis] = gear

        self.num_limbs += limbs2add
        self.limb_idx += limbs2add

    def _get_new_limbs(self, site, limbs2add, limb_params):
        # Types of sites to be made in the new limbs
        new_site_type = "growth_site"

        # Construct single limb
        limb, orient = self._construct_limb(
            self.limb_idx, site, new_site_type, limb_params
        )

        parent_idx = xu.name2id(site)
        if "torso" not in site.get("name") and self.num_limbs > 0:
            p_orient = self.limb_metadata[parent_idx]["orient"]
            if gu.is_same_orient(orient, p_orient) and "mid" in site.get("name"):
                return None, None, None

        if limbs2add == 1:
            return [limb], [site], [orient]

    def _get_orient_fromto(self, s):
        xyz12 = xu.str2arr(s)
        xyz = xyz12[3:] - xyz12[:3]
        (h, theta, phi) = gu.cart2sph(*xyz.tolist())
        return (h, theta, phi)

    def _choose_limb_params(self):
        orient = self._get_orient_fromto(self.cur_body.geoms[0].node.attrib['fromto'])
        limb_params = {
            "limb_radius": float(self.cur_body.geoms[0].node.attrib['size']), #su.sample_from_range(cfg.LIMB.RADIUS_RANGE),
            "limb_height": orient[0], #su.sample_from_range(cfg.LIMB.HEIGHT_RANGE),
            "limb_orient": orient,
            "limb_density": self.body_params["limb_density"],
        }

        joint_params = self._choose_joint_params()
        limb_params.update(joint_params)
        return limb_params

    def _choose_joint_params(self, joint_axis=None):
        return {
            "joint_axis": 'A', #joint_axis,
            "joint_axis3": xu.str2arr(self.cur_body.joints[0].node.attrib['axis']),
            "joint_range": [xu.str2arr(self.cur_body.joints[0].node.attrib['range'])], #joint_range,
            "gear": [float(self.cur_body.joints[0].actuator.node.attrib['gear'])], #gear,
        }

    def _choose_site(self):
        """Choose a free site, and how many limbs to add at the site."""

        # Choose a torso first, and then choose within the torso child elems

        torso_idx = self.growth_torso[0]
        torso = xu.find_elem(
            self.root, "body", "name", "torso/{}".format(torso_idx)
        )[0]
        site_name = self.body_name2site_name[self.cur_body.parent.name]
        site = xu.find_elem(torso, "site", "name", site_name)[0]
        limbs2add = 1
        return site, limbs2add

    def _is_symmetric(self, sim):
        """Check if current unimal is symmetric along BODY.SYMMETRY_PLANE."""

        # Get unimal center of mass (com)
        head_idx = sim.model.body_name2id(HEAD)
        unimal_com = sim.data.subtree_com[head_idx, :]
        # Center of mass should have zero component along axis normal to
        # cfg.BODY.SYMMETRY_PLANE.
        normal_axis = cfg.BODY.SYMMETRY_PLANE.index(0)
        return unimal_com[normal_axis] == 0

    def _new_contacts(self, sim, exclude_geom_pairs):
        """New contacts should only be between exclude_geom_pairs."""

        # exclude_geom_pairs will contain all parent child geom pairs being
        # added and if two geoms are added to the same parent, it will
        # contain that pair also. There should be no new contact other than
        # these pairs and previous such relations as it would indicate self-
        # intersection.
        exclude_geom_pairs = [
            tuple(
                sorted(
                    (
                        mu.mj_name2id(sim, "geom", geom1.get("name")),
                        mu.mj_name2id(sim, "geom", geom2.get("name")),
                    )
                )
            )
            for (geom1, geom2) in exclude_geom_pairs
        ]

        return exclude_geom_pairs

    def _update_joint_axis(self):
        sim = self.sim
        limbs = xu.find_elem(self.root, "body")
        limbs = [elem for elem in limbs if "torso" not in elem.get("name")]
        for limb in limbs:
            name = limb.get("name")
            # z-axis of geom frame always points in the direction end -> start
            # (or to --> from). Valid rotations are about x and y axis of the
            # geom frame.
            geom_frame = sim.data.get_geom_xmat(name).reshape(3, 3)
            joints = xu.find_elem(limb, "joint", child_only=True)
            for joint in joints:
                joint_name = joint.get("name")
                if joint_name[4] == "x":
                    curr_axis = geom_frame[:, 0]
                elif joint_name[4] == "y":
                    curr_axis = geom_frame[:, 1]
                joint.set("axis", xu.arr2str(curr_axis, num_decimals=4))

    def _align_joints_actuators(self):
        """Ensure that the joint order in body and actuators is aligned."""
        # Delete all gears
        motors = xu.find_elem(self.actuator, "motor")
        for motor in motors:
            self.actuator.remove(motor)

        # Add gears corresponding to joints
        joints = xu.find_elem(self.unimal, "joint", "class", "normal_joint")
        for joint in joints:
            name = joint.get("name")
            axis = name.split("/")[0][-1]
            gear = self.limb_metadata[xu.name2id(joint)]["gear"][axis]
            self.actuator.append(xu.actuator_elem(name, gear))

    def _add_sensors(self):
        sensor = self.root.findall("./sensor")[0]
        for s in sensor:
            sensor.remove(s)

        # Add imu sensors
        sensor.append(xu.sensor_elem("accelerometer", "torso_accel", "root"))
        sensor.append(xu.sensor_elem("gyro", "torso_gyro", "root"))
        # Add torso velocity sensor
        sensor.append(xu.sensor_elem("velocimeter", "torso_vel", "root"))
        # Add subtreeangmom sensor
        sensor.append(
            etree.Element("subtreeangmom", {
                "name": "unimal_am", "body": "torso/0"
            })
        )
        # Add touch sensors
        bodies = xu.find_elem(self.root, "body")
        for body in bodies:
            body_name = body.get("name").split("/")
            site_name = "{}/touch/{}".format(body_name[0], body_name[1])
            sensor.append(xu.sensor_elem("touch", body.get("name"), site_name))


    def save_image(self):
        # sim = mu.mjsim_from_etree(self.root)
        # sim.step()
        sim = self.sim
        frame = sim.render(
            cfg.IMAGE.WIDTH,
            cfg.IMAGE.HEIGHT,
            depth=False,
            camera_name=cfg.IMAGE.CAMERA,
            mode="offscreen",
        )
        # Rendered images are upside down
        frame = frame[::-1, :, :]
        imageio.imwrite(fu.id2path(self.id, "images"), frame)

    def _get_limb_r(self, name):
        elem = xu.find_elem(self.unimal, "geom", "name", name)[0]
        return float(elem.get("size"))

    def _exclude_permanent_contacts(self):
        # Enable filterparent
        flag = xu.find_elem(self.root, "flag")[0]
        flag.set("filterparent", str("enable"))

        sim = self.sim

        contact_pairs = mu.get_active_contacts(sim)
        for geom1, geom2 in contact_pairs:
            self.contact.append(xu.exclude_elem(geom1, geom2))

        flag.set("filterparent", str("disable"))

    def _before_save(self):
        self._align_joints_actuators()
        self._add_sensors()
        self._exclude_permanent_contacts()

    def save(self):
        self._before_save()
        xml_path = os.path.join(cfg.OUT_DIR, "xml", "{}.xml".format(self.id))
        xu.save_etree_as_xml(self.tree, xml_path)
        mutation_op = ""
        init_state = {
            "xml_path": xml_path,
            "contact_pairs": self.contact_pairs,
            "num_limbs": self.num_limbs,
            "limb_idx": self.limb_idx,
            "num_torso": self.num_torso,
            "torso_list": self.torso_list,
            "body_params": self.body_params,
            "limb_list": self.limb_list,
            "limb_metadata": self.limb_metadata,
            # "mirror_sites": self.mirror_sites,
            "dof": len(xu.find_elem(self.actuator, "motor")),
            "parent_id": self.parent_id,
            "mutation_op": mutation_op,
            "growth_torso": self.growth_torso,
        }
        save_path = os.path.join(
            cfg.OUT_DIR, "unimal_init", "{}.pkl".format(self.id)
        )
        fu.save_pickle(init_state, save_path)


if __name__ == '__main__':
    from khrylib.robot.xml_robot import Robot

    model_xml_file = f'assets/mujoco_envs/ant6.xml'
    robot_cfg = {
        'param_mapping': 'sin',
        'no_root_offset': True,
        'axis_vertical': True,
        'body_params': {'offset': {'type': 'xy', 'lb': [-0.5, -0.5], 'ub': [0.5, 0.5]}},
        'joint_params': {},
        'geom_params': {'size': {'lb': 0.03, 'ub': 0.1}, 'ext_start': {'lb': 0.0, 'ub': 0.2}},
        'actuator_params': {'gear': {'lb': 20, 'ub': 400}}
    }

    cur_robot = Robot(robot_cfg, xml=model_xml_file)

    a = Morphology('1', cur_robot)
    # a._before_save()
    print(a)