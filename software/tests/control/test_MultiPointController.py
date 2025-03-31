import tests.control.gui_test_stubs as gts
import pytest


def test_multi_point_controller_image_count_calculation(qtbot):
    mpc = gts.get_test_multi_point_controller()

    all_configuration_names = [
        config.name
        for config in mpc.channelConfigurationManager.get_configurations(mpc.objectiveStore.current_objective)
    ]
    nz = 2
    nt = 3
    assert len(all_configuration_names) > 0
    all_config_count = len(all_configuration_names)

    mpc.set_NZ(nz)
    mpc.set_Nt(nt)
    mpc.set_selected_configurations(all_configuration_names[0:1])
    mpc.scanCoordinates.clear_regions()

    assert mpc.get_acquisition_image_count() == 0

    # Add a single region with 1 fov
    # NOTE: If the coordinates below aren't in the valid range for our stage, it silently fails to add regions.
    x_min = mpc.stage.get_config().X_AXIS.MIN_POSITION + 0.01
    y_min = mpc.stage.get_config().Y_AXIS.MIN_POSITION + 0.01
    z_mid = (mpc.stage.get_config().Z_AXIS.MAX_POSITION - mpc.stage.get_config().Z_AXIS.MIN_POSITION) / 2.0
    mpc.scanCoordinates.add_flexible_region(1, x_min, y_min, z_mid, 1, 1, 0)

    assert mpc.get_acquisition_image_count() == (nt * nz * 1 * 1)

    # Add 9 more regions with a single fov
    for i in range(1, 10):
        x_st = x_min + i
        y_st = y_min + i
        mpc.scanCoordinates.add_flexible_region(i + 2, x_st, y_st, z_mid, 1, 1, 0)

    assert mpc.get_acquisition_image_count() == (nt * nz * 10 * 1)

    # Select all the configurations
    mpc.set_selected_configurations(all_configuration_names)
    assert mpc.get_acquisition_image_count() == (nt * nz * 10 * all_config_count)

    # Add a multiple FOV region with 5 in each of x and y dirs.
    mpc.scanCoordinates.add_flexible_region(123, x_min + 11, y_min + 11, z_mid, 5, 5, 0)

    assert mpc.get_acquisition_image_count() == (nt * nz * (10 + 25) * all_config_count)
