# Oral Version for Recording

This version is intentionally more conversational and easier to say out loud. It follows the same slide order as the presentation, but the phrasing is shorter and more natural.

## Timing target

- Target total: **12:45 to 13:10**
- **Jordan**: around **6 min 20 s**
- **Gustavo**: around **6 min 25 s**

## Jordan

### Slide 1 -- Title

Hello, we are Jordan Ikukele Bomolo Mia and Gustavo Garzo Chumilla, and this is our mini-project on snow pole detection using driving images from Trondheim.

### Slide 2 -- Outline

I will start with the problem motivation and the exploratory data analysis, and then Gustavo will continue with the models, the results, and the final discussion.

### Slide 3 -- Background & Motivation

The motivation behind this project is winter driving. In snowy conditions, normal road cues like lane markings or road boundaries can become very hard to see. Because of that, snow poles become a very useful visual reference for road localisation. So from a computer vision point of view, this becomes a one-class object detection problem, but still a difficult one, because the objects are thin, small, and often far from the camera.

### Slide 4 -- Background & Motivation -- Visual Examples

This slide already shows that the two datasets do not look exactly the same. The `Road_poles_iPhone` images are generally brighter and cleaner, while `roadpoles_v1` looks darker and visually harder. So even before doing the EDA, we already had a hint that the two leaderboards should be treated separately.

### Slide 5 -- Approach / Strategy

Because of that, we did not want to jump straight into training models. We first approached the project as a dataset understanding problem. The idea was to analyse both datasets properly, understand what really makes the task difficult, and then use that to justify the choices made later in the modelling part.

### Slide 6 -- Approach / Strategy -- Key Reasoning

In practice, the EDA was built around three questions. First, are the two datasets actually similar enough to compare directly? Second, is the task hard because of object density, object size, geometry, or scene structure? And third, which of these factors is most likely to explain the final performance differences?

### Slide 7 -- Data Analysis -- Dataset Overview

According to this chart, the first big difference is simply dataset size. `Road_poles_iPhone` is much larger, with `942` training images and `261` validation images, while `roadpoles_v1` only has `322` training and `92` validation images. So one dataset gives us much more supervision than the other. That is important, because it means training on the iPhone dataset should be more stable, and it also means the comparison between the two benchmarks should not be interpreted as fully equal.

### Slide 8 -- Data Analysis -- Object Geometry

This is probably the most informative EDA figure in the presentation. It compares three things: how large the boxes are, how elongated the poles are, and how many boxes appear per image. What we see is that `roadpoles_v1` is harder geometrically. The poles are smaller and also much thinner relative to their height. At the same time, both datasets have only around `1.2` poles per image, so the problem is clearly not crowding. The challenge is accurate detection of very thin roadside objects.

### Slide 9 -- Data Analysis -- Object Scale

This histogram supports exactly the same conclusion. Most bounding boxes are packed very close to zero, which means that in both datasets the poles occupy only a tiny part of the image. So the task is fundamentally a small-object detection problem. Also, `roadpoles_v1` is shifted slightly toward even smaller objects, which again suggests that it is the harder of the two benchmarks. This is why later on resolution becomes such an important training choice.

### Slide 10 -- Data Analysis -- Spatial Distribution

The heatmaps are also useful, because they show that the poles are not appearing randomly across the image. Instead, their centres are concentrated in a relatively narrow band. That makes sense physically, because the poles are placed along the road, so they tend to appear in structured positions in the frame.

### Slide 11 -- Data Analysis -- Spatial Distribution Insights

What that means in practice is that failures are less likely to come from clutter or heavy overlap, and more likely to come from poor visibility and inaccurate localisation on small thin objects. So with that EDA in mind, we can now understand the motivation behind the model choices.

### Slide 22 -- Key Learning Points

For us, there were three main takeaways. First, image resolution matters a lot in small-object detection, which is completely consistent with the EDA. Second, building the Faster R-CNN pipeline helped us understand two-stage detection much better in practice. And third, a lot of real deep learning work is actually engineering work: handling dataset formats, paths, YAML files, and framework-specific label conversions.

### Slide 24 -- Summary

To summarize, both models were able to detect poles very reliably at `mAP@50`, but YOLOv8s gave the best overall balance between accuracy, localisation quality, and deployment potential. So our main conclusion is that for this task, the most practical model is YOLOv8s, and the most important lever is preserving enough image resolution for very small objects.

## Gustavo

### Slide 12 -- Overview -- Two Architectures

To study the problem from two angles, we used two different detector families: YOLOv8s as a fast one-stage detector, and Faster R-CNN with a ResNet-50 FPN backbone as a more classical two-stage detector.

### Slide 13 -- Model 1 -- YOLOv8s Architecture

Starting with YOLOv8s, the key motivation was speed plus enough model capacity for small objects. Since the poles are only around two percent of the image width, we wanted more capacity than the nano version. At the same time, YOLO is attractive because it processes the whole image in a single forward pass, so it is much more realistic for real-time deployment on a vehicle.

### Slide 14 -- Model 1 -- YOLOv8s Training Config

This slide shows the training setup we used for YOLO. The main points are `150` epochs, image size `1280`, patience `30`, and several augmentations such as HSV variation, flipping, mosaic, scaling, and translation. The most important idea here is that we wanted to preserve as much useful detail as possible for these small poles.

### Slide 15 -- Model 2 -- Faster R-CNN Architecture

For the second model, we used Faster R-CNN. This is a two-stage detector: first it proposes candidate regions, and then it classifies and refines them. We paired it with a ResNet-50 FPN backbone, which is useful because FPN represents the image at multiple scales. That is especially relevant here, since the poles can appear at very different distances from the camera.

### Slide 16 -- Model 2 -- Faster R-CNN Training Config

One practical challenge with Faster R-CNN was data formatting. Our labels were in YOLO format, using normalized centre coordinates, but Torchvision expects absolute corner coordinates. So we had to convert every box before training. We also used training augmentations such as color jitter and horizontal flips.

### Slide 17 -- Results -- Validation Set Metrics

According to this table, both models did very well at `mAP@50`, with values above `0.99`, so both of them can generally find the poles. The bigger difference appears in `mAP@0.5:0.95`, where YOLO reaches `0.788` and Faster R-CNN reaches `0.626`. YOLO also shows much stronger recall. So overall, YOLO gives more consistent and tighter detections.

### Slide 18 -- Results -- Leaderboard

On the hidden test server, the scores are lower, which is expected. YOLO got around `69%` on the `Road poles iPhone` leaderboard, while Faster R-CNN got around `66%` on the `roadpoles_v1` leaderboard. This suggests that generalization to unseen conditions is still difficult.

### Slide 19 -- Results -- mAP@50 vs mAP@0.5:0.95 Gap

This slide explains why the stricter metric matters. Even though both models score extremely high at `mAP@50`, YOLO keeps much more of that performance when we require tighter overlap. Faster R-CNN drops much more. So the real difference is not whether the models see the poles at all, but how precisely they localize them.

### Slide 20 -- Discussion -- Why YOLO Outperformed Faster R-CNN

There are a few likely reasons for that. YOLO was trained under more favorable conditions: a larger dataset, higher image resolution, and stronger augmentation. So even though Faster R-CNN is theoretically appealing for small objects, in this project YOLO had the practical advantage.

### Slide 21 -- Discussion -- Limitations & Future Work

This also tells us where the project could be improved. The comparison is not perfectly balanced, because Faster R-CNN was trained on the smaller dataset. We also see some domain shift on the hidden test server. So future work could include higher resolution for Faster R-CNN, pseudo-labeling with the unlabeled data, or test-time augmentation and ensembles.

### Slide 23 -- Sustainability -- Compute Energy

Finally, for sustainability, we estimated the compute cost based on training on an RTX 4090 with a `450` watt TDP. Assuming about `4` hours of total training, that gives roughly `1.8` kilowatt-hours. For a project of this scale, and especially one based on fine-tuning rather than training from scratch, this remains a relatively modest energy footprint.

### Slide 25 -- Thank you for listening

No narration here. Just leave a short pause at the end so the video closes cleanly.
