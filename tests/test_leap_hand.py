import jax
import mujoco
import numpy as np
from mujoco import mjx

from cube_ppo import ROOT


def test_model():
    """Make sure we can load the hand into mujoco and mjx."""
    model_file = ROOT + "/envs/leap_hand/leap_hand.xml"

    # Test the mujoco model
    mj_model = mujoco.MjModel.from_xml_path(model_file)
    mj_data = mujoco.MjData(mj_model)

    assert mj_model.nq == 16
    assert mj_model.nv == 16
    assert mj_model.nu == 16
    assert mj_data.qpos.shape == (16,)
    assert mj_data.qvel.shape == (16,)
    assert mj_data.ctrl.shape == (16,)

    old_q = mj_data.qpos.copy()
    mujoco.mj_step(mj_model, mj_data)
    new_q = mj_data.qpos.copy()

    assert np.all(old_q != new_q)

    # Test translating it to mjx
    mjx_model = mjx.put_model(mj_model)
    assert isinstance(mjx_model, mjx.Model)

    mjx_data = mjx.put_data(mj_model, mj_data)
    assert isinstance(mjx_data, mjx.Data)


if __name__ == "__main__":
    test_model()
