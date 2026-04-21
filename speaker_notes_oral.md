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

This histogram supports exactly the same conclusion. Most bounding boxes are packed very close to zero, which means that in both datasets the poles occupy only a tiny part of the image. So the task is fundamentally a small-object detection problem. Also, `roadpoles_v1` is shifted slightly toward even smaller objects, which again suggests that it is the harder of the two benchmarks.

### Slide 10 -- Data Analysis -- Spatial Distribution

The heatmaps are also useful, because they show that the poles are not appearing randomly across the image. Instead, their centres are concentrated in a relatively narrow band. That makes sense physically, because the poles are placed along the road, so they tend to appear in structured positions in the frame.

### Slide 11 -- Data Analysis -- Spatial Distribution Insights

What that means in practice is that failures are less likely to come from clutter or heavy overlap, and more likely to come from poor visibility and inaccurate localisation on small thin objects. So with that EDA in mind, we can now understand the motivation behind the model choices.

### Slide 22 -- Key Learning Points

For us, there were three main takeaways. First, the EDA was actually predictive, because it correctly suggested that `roadpoles_v1` would be harder. Second, building the Faster R-CNN pipeline helped us understand two-stage detection much better in practice. And third, a lot of real deep learning work is actually engineering work: handling dataset formats, paths, YAML files, and framework-specific label conversions.

### Slide 24 -- Summary

To summarize, the strongest validated results in our notebooks came from Faster R-CNN, and those results were clearly better on `Road_poles_iPhone` than on `roadpoles_v1`. So our main conclusion is that Faster R-CNN was the best-performing model family in this project, and that the EDA explained the difficulty difference between the two datasets quite well.

## Gustavo

### Slide 12 -- Overview -- Two Architectures

To study the problem from two angles, we used two different detector families: YOLO as a fast single-stage baseline, and Faster R-CNN as a more classical two-stage detector. Both model families were used on both datasets.

### Slide 13 -- Model 1 -- YOLO Baseline Architecture

Starting with YOLO, the idea here was to have a lightweight baseline. It is simple to train, fast to run, and naturally relevant for deployment because it processes the whole image in one forward pass.

### Slide 14 -- Model 1 -- YOLO Training Config

This slide shows the baseline YOLO setup from the notebooks. We used `yolov8n`, trained it for `100` epochs, and used image size `640`. So this part of the project was mainly there to give us a fast reference model.

### Slide 15 -- Model 2 -- Faster R-CNN Architecture

For the second model, we used Faster R-CNN. This is a two-stage detector: first it proposes candidate regions, and then it classifies and refines them. That is especially relevant here because the poles are thin and often far away, so careful localisation matters a lot.

### Slide 16 -- Model 2 -- Faster R-CNN Training Config

One practical challenge with Faster R-CNN was data formatting. Our labels were in YOLO format, using normalized centre coordinates, but Torchvision expects absolute corner coordinates. So we had to convert every box before training. In the notebooks, Faster R-CNN also used `AdamW`, cosine annealing, and `100` epochs.

### Slide 17 -- Results -- Validation Set Metrics

According to this table, the strongest completed validation results in the notebooks are the Faster R-CNN results on both datasets. The main point is that the same model performs better on `Road_poles_iPhone` than on `roadpoles_v1`. That fits the EDA very well, because the iPhone dataset looked easier from the start.

### Slide 18 -- Results -- Leaderboard

On the hidden test server, the scores are lower, which is expected. Faster R-CNN was the stronger final submission on both leaderboards, with about `69%` on `Road poles iPhone` and about `66%` on `roadpoles_v1`. This suggests that generalization to unseen conditions is still difficult.

### Slide 19 -- Results -- Same Model on Two Datasets

This slide explains the dataset gap more clearly. `mAP@50` is almost the same on both datasets, so Faster R-CNN can find the poles in both. But the stricter metric drops much more on `roadpoles_v1`, which tells us that the real problem there is tight localisation.

### Slide 20 -- Discussion -- Why Faster R-CNN Performed Best

There are a few likely reasons for that. Faster R-CNN gives the model a proposal stage plus box refinement, which is useful for thin objects. Its multi-scale representation also fits the problem well. And finally, the results are coherent with the EDA, which makes the whole story more convincing.

### Slide 21 -- Discussion -- Limitations & Future Work

This also tells us where the project could be improved. `roadpoles_v1` is still clearly the harder dataset, and we also see domain shift on the hidden test server. So future work could include better sweeps for both models, pseudo-labeling with unlabeled data, or test-time augmentation and ensembles.

### Slide 23 -- Sustainability -- Compute Energy

Finally, for sustainability, we estimated the compute cost based on training on an RTX 4090 with a `450` watt TDP. Assuming about `4` hours of total training, that gives roughly `1.8` kilowatt-hours. For a project of this scale, and especially one based on fine-tuning rather than training from scratch, this remains a relatively modest energy footprint.

### Slide 25 -- Thank you for listening

No narration here. Just leave a short pause at the end so the video closes cleanly.
