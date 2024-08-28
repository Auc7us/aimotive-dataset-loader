from enum import Enum

class BulbColor(str, Enum):
    red = "red"
    yellow = "yellow"
    green = "green"
    unknown = "unknown"

trafficlightcolor2int = {
    # v-3line
    (BulbColor.red, BulbColor.unknown, BulbColor.unknown): 0,
    (BulbColor.red, BulbColor.yellow, BulbColor.unknown): 1,
    (BulbColor.unknown, BulbColor.yellow, BulbColor.unknown): 2,
    (BulbColor.unknown, BulbColor.unknown, BulbColor.green): 3,
    (BulbColor.unknown, BulbColor.unknown, BulbColor.unknown): 4
}

trafficlightcolor_to_bgr = {
    # v-3line
    (BulbColor.red, BulbColor.unknown, BulbColor.unknown): (0, 0, 255),
    (BulbColor.red, BulbColor.yellow, BulbColor.unknown): (0, 200, 255),
    (BulbColor.unknown, BulbColor.yellow, BulbColor.unknown): (0, 255, 255),
    (BulbColor.unknown, BulbColor.unknown, BulbColor.green): (0, 255, 0),
    (BulbColor.unknown, BulbColor.unknown, BulbColor.unknown): (0, 0, 0),
}

def get_lightcolors(box):
    return tuple(x["color"] for x in box["ObjectMeta"]["Lights"])
