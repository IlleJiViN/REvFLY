# REvFLY presentation and demonstration guide

## Four-minute presentation

**Problem (40 seconds)**

“A fruit fly simulator takes muscle inputs and computes how the leg moves. I built an engine that solves the inverse problem. You give it the desired joint motion, and it estimates the motor firing rates needed to produce it.”

**Engine (50 seconds)**

Specify 420 ms of target angles for the seven axes of the left front leg. An MLP predicts firing rates for 15 motor channels. The engine converts these rates to muscle drives and replays them in FlyMimic with MuJoCo. It measures the difference between the target and the simulated trajectory and can refine the rates for that target. The control interface displays the trajectories and a top-view video.

**Evaluation (75 seconds)**

We generated nine targets from three synthetic motion families held out from training and model selection. The same physics model produced all nine trajectories. We gave the same forward search and 30-second budget to two initializations, a baseline rate waveform shared by all targets and rates predicted by the MLP for each target. Mean error fell from **10.412° to 6.056°**, and the number of targets above the error threshold fell from **9/9 to 3/9**. A run exceeded the threshold when its RMSE over all seven axes was greater than 0.1 rad (5.7296°). [Comparison plot and results](../reports/unseen_motion_families_v2/README.md)

**Scope and limitations (45 seconds)**

In the gradient-based comparison on the same targets, 15 of 18 runs stopped at the gradient check and all runs exceeded the error threshold. The figures above come from the follow-up forward search. The experiments cover seven axes of the simulated front leg with the body fixed. The engine maps target trajectories to comparable motor commands and checks their tracking error in the same simulator.

## Live demonstration

Complete the installation before the talk. Run these commands in the installed environment. [Installation instructions](../README.md#installation)

```bash
revfly run-multijoint --goal-family swing_scaled --scale 0.92 \
  --mode fast --output ./presentation-swing --render
revfly serve --host 127.0.0.1 --port 8505 --ui \
  --model-id 20261115_v2_fastbridge_lowenergy
```

In Control Studio, select the seven-axis left front leg mode and the swing target. Compare the target and replay curves one axis at a time. Show the firing rate for one of the 15 muscles and the top-view video. The swing preset demonstrates the workflow. The nine held-out targets were evaluated in the benchmark. If live rendering is impractical, play the video saved in `presentation-swing`. [Local Control Studio](http://127.0.0.1:8505/)

## Questions and answers

| Question | Answer |
| --- | --- |
| Did you use a GNN to generate the motion? | We tested a GNN for predicting motor-neuron responses from sensory inputs. The inverse controller presented here uses a ResidualNorm MLP and MuJoCo. |
| What does 6.056° measure? | It is the mean of the nine target RMSEs, each computed across seven axes and 420 ms. Peak error on an individual axis can be larger. |
| Is there a unique firing-rate solution? | Different commands can reproduce similar motion. We independently replayed the selected command and recorded its error. |
| Did you move a live fruit fly? | The results come from FlyMimic simulations. A live experiment would require separate validation of the stimulation needed to produce the firing rates and the resulting movement. |
| What did REvFLY add to FlyMimic? | FlyMimic provides the musculoskeletal model used to simulate motion. REvFLY adds inverse prediction of firing rates from target trajectories, refinement and replay verification, a Python API, and a local control interface. |
