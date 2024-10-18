import jax
import mujoco
import numpy as np
from mujoco import mjx

from cube_ppo import ROOT
from cube_ppo.envs.leap_hand.cube_rotation_env import CubeRotationEnv


def test_model():
    """Make sure we can load the hand into mujoco and mjx."""
    model_file = ROOT + "/envs/leap_hand/scene.xml"

    # Test the mujoco model
    mj_model = mujoco.MjModel.from_xml_path(model_file)
    mj_data = mujoco.MjData(mj_model)

    assert mj_model.nq == 23
    assert mj_model.nv == 22
    assert mj_model.nu == 16
    assert mj_data.qpos.shape == (23,)
    assert mj_data.qvel.shape == (22,)
    assert mj_data.ctrl.shape == (16,)

    old_q = mj_data.qpos.copy()
    mujoco.mj_step(mj_model, mj_data)
    new_q = mj_data.qpos.copy()

    assert not np.all(old_q == new_q)

    # Test translating it to mjx
    mjx_model = mjx.put_model(mj_model)
    assert isinstance(mjx_model, mjx.Model)

    mjx_data = mjx.make_data(mjx_model)
    assert isinstance(mjx_data, mjx.Data)

    old_qx = mjx_data.qpos.copy()
    mjx_data = jax.jit(mjx.step)(mjx_model, mjx_data)
    new_qx = mjx_data.qpos.copy()

    assert np.allclose(old_q, old_qx)
    assert np.allclose(new_q, new_qx)


def test_env():
    """Make sure we can create and step the cube rotation environment."""
    rng = jax.random.PRNGKey(0)
    env = CubeRotationEnv()

    assert env.cube_position_sensor != -1
    assert env.cube_orientation_sensor != -1
    assert env.cube_linvel_sensor != -1
    assert env.cube_angvel_sensor != -1

    rng, reset_rng = jax.random.split(rng)
    state = env.reset(reset_rng)
    assert state.obs.shape == (44,)

    rng, action_rng = jax.random.split(rng)
    action = jax.random.uniform(action_rng, (16,))
    state = env.step(state, action)

    assert state.obs.shape == (44,)
    assert state.reward.shape == ()
    assert state.done.shape == ()
    assert state.metrics["reward"].shape == ()

    assert state.reward < 0.0
    assert state.reward == state.metrics["reward"]


if __name__ == "__main__":
    # test_model()
    test_env()
