from .pick_place import PickPlace, PickPlaceOpen, PickPlaceOpenSuboptimal
from .drawer_open import DrawerOpen
from .grasp import Grasp, GraspTransfer, GraspTransferSuboptimal
from .place import Place
from .button_press import ButtonPress
from .open_grasp import DrawerOpenGrasp
from .drawer_open_transfer import (
    DrawerOpenTransfer,
    DrawerOpenTransferSuboptimal
)
from .drawer_close_open_transfer import (
    DrawerCloseOpenTransfer,
    DrawerCloseOpenTransferSuboptimal
)

policies = dict(
    grasp=Grasp,
    grasp_transfer=GraspTransfer,
    grasp_transfer_suboptimal=GraspTransferSuboptimal,
    pickplace=PickPlace,
    pickplace_open=PickPlaceOpen,
    drawer_open=DrawerOpen,
    drawer_open_grasp=DrawerOpenGrasp,
    button_press=ButtonPress,
    drawer_open_transfer=DrawerOpenTransfer,
    place=Place,
    drawer_close_open_transfer=DrawerCloseOpenTransfer,
)

suboptimal_polices = dict(
    drawer_open_transfer_suboptimal=DrawerOpenTransferSuboptimal,
    drawer_close_open_transfer_suboptimal=DrawerCloseOpenTransferSuboptimal,
    pickplace_open_suboptimal=PickPlaceOpenSuboptimal,
)

policies.update(suboptimal_polices)
