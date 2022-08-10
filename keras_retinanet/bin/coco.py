from pycocotools.coco import COCO
import numpy as np
import cv2
import os
import sys
import argparse


def count_categories(args):
    coco_dataset_path = "../DATASET/PALM/ALL/annotations"
    coco_path = os.path.join(coco_dataset_path, "instances_{instance}.json".format(instance=args.instance))
    print("coco_path", coco_path)
    coco = COCO(coco_path)
    
    # ============================================================================

    ann_ids = coco.getAnnIds(iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    anns = [ann for ann in anns if len(ann['segmentation']) == 0]
    print("num of annotations with more than one polygan:", len(anns))    # 3522
    objects = {"Object": 0, "Abnormal": 0, "Janjang Kosong": 0,
            "Kurang Masak": 0, "Masak": 0, "Mentah": 0, "Terlalu Masak": 0}
    for i, ann in enumerate(anns):
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        segs = ann["segmentation"]
        bbox = np.array(ann["bbox"])
        bbox[2:4] = bbox[0:2] + bbox[2:4]
        print("BBox[{}] (xyxy):".format(i), bbox.tolist())

        image_info = coco.loadImgs(image_id)
        image_path = image_info[0]["file_name"]
        category = coco.loadCats(category_id)[0]["name"]
        objects[category] += 1

        # [0] is required, always return a list
        image_path = os.path.join(
            coco_dataset_path, "images", "test", image_path)
        print(image_path)
        # /export/public/MS-COCO-2017/val2017/000000061108.jpg

        image = cv2.imread(image_path)
        segs = [np.array(seg, np.int32).reshape((1, -1, 2))
                for seg in segs]

        for seg in segs:
            cv2.drawContours(image, seg, -1, (0, 255, 0), 2)
        # third aug -1 means draw all contours in 3-D array, Or
        # for seg in segs: cv2.fillPoly(image, segm, (0,255,0))
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

        # cv2.putText(image, category, (bbox[0], bbox[1] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        # cv2.imshow(category, image)
        # cv2.waitKey(0)

    print(objects)

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='COCO script for counting object categories from images.')

    parser.add_argument(
        '--instance',    help='Dataset instance (train/valid/test)', default='train')

    return parser.parse_args(args)

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    count_categories(args)
    
if __name__ == '__main__':
    main()

# ============================================================================


# ann_ids = coco.getAnnIds(iscrowd=True)
# anns = coco.loadAnns(ann_ids)

# import pycocotools.mask as mask

# for ann in anns:
#     image_id = ann["image_id"]
#     segm = ann['segmentation']
#     bbox = ann['bbox']

#     assert sum(segm['counts']) == segm['size'][0] * segm['size'][1]

#     # Draw RLE label
#     label = np.zeros(segm['size'], np.uint8).reshape(-1)
#     ids = 0
#     value = 0
#     for c in segm['counts']:
#         label[ids: ids+c] = value
#         value = not value
#         ids += c
#     label = label.reshape(segm['size'], order='F')
#     # order='F' means Fortran memory order
#     cv2.imshow("RLE label", label*255)
#     cv2.waitKey()

#     print('Encoded RLE:', mask.frPyObjects(segm, *segm['size']))
#     """
#     Encoded RLE: {'size': [240, 320], 'counts': b'`824200N5OI?Y1o3]OfK@;T1m3]OkK\\O9W1k3^OnKXO7[1`3gNTLg0c0ZOYOX1n3IhL>V3DiL=U3FiL;V3HhL8W30cLO[34dLLZ36hLHX38hLHX39fLHZ38eLH\\38cLI]38bLG_3:`LF`3<ZLRN1a1f3U20]Oc0M3N2M3M3M3N2M3J6F:K5L4M3N2M3N2VOWJKk54ZJDj5:g0O1O100O10000000000000000O1000UJCc4=]KEa4;_KF`49aKI]48bKI]47cKK[45eKLZ44fKMY43fK0X40fK5W4LfKa0o3@nKc0Q4_OkKd0T4]OiKf0V4[OfKi0Y4XOcKk0]4VO^Ko0a4RO^Ko0a4QO^KP1b4QO]KP1b4PO]KQ1c4PO\\KP1d4PO\\KQ1c4oN\\KR1d4oN[KQ1e4oN[KQ1e4POYKQ1g4oNYKQ1g4j02J6O1kNWKCi4<ZK]OG]Oo4U1\\KZOHAl4U1^KjNF62Jj4V1gKnNb4m0dKQO]4n0S10kJoNk3Q1SLQOm3o0QLSOo3m0oKXOn3j0PKoNk0>o3h0PLWOQ4i0nKXOR4h0nKWOS4;WK_Oe06U4;XK]O5J14F1m4d0XKCLF`02]4d0XKL:_O_4e0WKL:^O`4f0VKM7]Oe4c0WK00@j4<\\K3H@n4=_KOBAR5?_KN^OAV5`0_KNh42ZKKg44]KFf49Y1O1O1O1O1O1O1000000O10000000000O10000001O00001O00001O001O2N1O2N1O2N4L8ZIVOV6R10000000001O001O0000]OQJFn5:VJBj5>XJ@h5`0YJ@f5`0[J_Oe5a0\\J_Oc5a0^J^Ob5b0_J^O`5b0`J_O_5a0bJ^O^5e0cJXO\\5j0d02N1O5dISOk5[1N1O001O001O1O001O00001O0000000000000000000000000000UOXJNh5MbJN^5OhJMY51kJMU51mJNT50nJ0R50oJNR52PKLP54V1002N1O2N1O2N2hI[OJ0X5f0fJXOJ560X5f0eJ@1LY5Y1bJkN]5e1O2N2N@hJiNW5g0ZKTO@HU5S1\\KTOj4k0WKTOj4k0WKUOi4k0WKTOj4m0UKSOk4m0UKUOi4k0XKVOf4j0P11O3M4L5oJaNg3_1XLcNg3^1WLdNi3[1VLfNk3[1PLiNP4Y1mKhNR4[1kKeNV4]1hKcNX4]1gKcNZ4]1eKcN\\4]1cKdN]4\\1aKeNa4[1]KeNd4`1UKaNm4P23M2N1O2N1O2N3M3M3M1O2M2O1N3N1N3N1N2O1N2N2N2N2N3L3N3L5J7HVi9'}
#     """

#     image_info = coco.loadImgs(image_id)
#     image_path = image_info[0]["file_name"]
#     image_path = os.path.join(coco_dataset_path, "val2017", image_path)
#     print("image path (crowd label)", image_path)
#     # /export/public/MS-COCO-2017/val2017/000000448263.jpg
#     image = cv2.imread(image_path)
#     cv2.imshow("RLE label", image)
#     cv2.waitKey()

#     break
