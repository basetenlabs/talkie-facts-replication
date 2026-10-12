"""The data and sample order of every write match the original run (talkie-facts job w7xzjo3)."""

from talkie_facts_replication import data
from talkie_facts_replication.writes import json_digest, sample_order, stage_seed

# Per write: hash of its chat-formatted training rows, and the first eight items of its 96-step sample order.
ORIGINAL = {
    1: ("9a01b1085117e2adb87a61662f31230db060424f31a341fe98f9dfe3bd10e30b", [16, 17, 6, 8, 21, 12, 1, 23]),
    2: ("cab7c1065e39a823456d91db3b75ae8b3c3e7587730e366a032abff638080643", [13, 1, 18, 8, 22, 12, 2, 19]),
    3: ("629497cf433b55cd31b25bd7929602199fd40406aaa86b1ebedb2aea32f8dbe7", [13, 20, 18, 8, 9, 6, 16, 1]),
    4: ("512ebadeabde091f7d46c7d2fbc558ecaa44dc9eb6a98ce7d1ad0e2d1e714b39", [18, 2, 20, 1, 3, 5, 11, 15]),
    5: ("8d1a40892cbcaf0ab0b6809264305c5e82008e90c0ea02d828b8bfe53d747fbe", [14, 8, 17, 6, 12, 10, 4, 9]),
    6: ("9e03c3d7572750f5a224baff9a6f72864c5227023c5252e86fb5dd3cbd3b19fd", [13, 12, 2, 8, 18, 9, 11, 15]),
    7: ("f295f87277506b103518740db6c5b1a8568627a260d8c28c43c28ffdb4e106e5", [9, 17, 13, 12, 7, 6, 8, 22]),
    8: ("8ea737a8a0da62ac4ae9a1411cd3fe1b134d1ddad152046bfe5146f1a04fa6a9", [1, 7, 6, 13, 19, 9, 21, 5]),
    9: ("d2213877e6836b76a2cb134be21ec82274ad686f267f09ed96989afe25f3cef0", [18, 9, 20, 8, 12, 17, 6, 22]),
    10: ("5cd87ed26c1d9695dfdeb112643d976b2c9cc2a50c2c4a647faa5695fca59f48", [3, 2, 17, 23, 7, 8, 4, 21]),
    11: ("54f43f127e4a0b680c144f23cee4e5b0831dacd37795b7ebd6b67ffd98e500d1", [11, 21, 2, 6, 8, 22, 5, 13]),
    12: ("70afb6d8160b99b33531beab39b5e0bc47872038209026f93c5ada94fe49afe5", [3, 18, 22, 6, 16, 5, 17, 7]),
    13: ("0a613d64f74addf55975dbd4194b4d69a3fd6899da332ddd31506132c2ee7acf", [20, 12, 15, 19, 2, 8, 1, 5]),
    14: ("7ba46e4d97f846ea4d1808a38007659f6922c523628bb440ce47ca5a381cc45d", [7, 23, 20, 13, 5, 0, 22, 21]),
    15: ("e1bd1570ef87de4dc32ffce3a72db5987c4b9013c7cf49682e9fd5e627e0bf1c", [19, 2, 4, 22, 5, 14, 3, 12]),
    16: ("20981dc8396ae462715e7e9ce390ad7a97dca95083e3f148e4e17d5f1a5e76a6", [15, 14, 2, 0, 23, 13, 20, 7]),
    17: ("36e79128e03e3238d44b5a1b3972517f0321dda3f8ad7c7edfc0e65955f4246d", [22, 6, 16, 2, 14, 3, 1, 9]),
    18: ("7464c0dce30a7300a64e23c21f96ffba382d9a8eb4b903014accb88c5eb30180", [8, 6, 9, 13, 12, 10, 17, 18]),
    19: ("6da49e96f19bbb51e6ca7946f8cc55959c4250cd4986d11105a87d9173c12726", [4, 17, 23, 21, 10, 3, 0, 9]),
    20: ("820c419d79de45dc58286385e827507914fb59f6087a546c9b2b99a8b077f608", [15, 0, 11, 8, 16, 10, 5, 9]),
}


def test_every_write_trains_on_the_original_rows_in_the_original_order():
    for fact in data.facts():
        stage = fact["write_position"]
        rows = [dict(id=r["id"], fact_id=fact["id"], prompt=r["prompt"], completion=r["completion"])
                for r in data.training_rows(fact["id"])]
        digest, order = ORIGINAL[stage]
        assert json_digest(rows) == digest
        assert sample_order(len(rows), 96, stage_seed(stage, "data"))[:8] == order
