# RGB-D Method Section – Draft (hypothetical implementation; not yet validated)

> This text describes the research prototype delivered in this package and is **not a statement of completed experiments**. Before inclusion in a journal revision, ensure that each described mechanism matches the final implemented and trained model. Cite Wang et al. (CVPR 2026) explicitly.

## 3.1 Overview

We study RGB-D salient object detection under imperfect depth measurements. Unlike methods developed for unaligned RGB-thermal inputs, we assume approximately registered RGB-D pairs and focus on the local reliability of depth measurements rather than cross-modal geometric registration. Our proposed research prototype contains an RGB-centric hierarchical encoder, a lightweight depth-specific convolutional encoder, a prior-guided depth uncertainty module (PG-DUM), and a difficulty-aware progressive fusion module (DAPF). Rather than inserting an uncertainty-alignment/fusion pair at every encoder stage, depth reliability is estimated at the two shallow stages; the intermediate stages use selective fusion, while the deepest representation preserves the RGB semantic pathway.

## 3.2 Prior-Guided Depth Uncertainty Modeling

Let R and D denote the RGB image and its registered depth counterpart, with M the binary depth validity mask. We extract RGB edge magnitude Er, depth gradient magnitude Ed, and an edge discrepancy term DeltaE = |Er - Ed|. A lightweight prior encoder transforms [Er, Ed, DeltaE, M] into a structural prior P. At scale l, the depth-specific encoder produces Fd^l and the RGB encoder produces Fr^l. The module predicts a conditional depth reconstruction distribution:

q_phi(d_l | Fr^l, Fd^l, P_l) = N(mu_l, exp(s_l)),

where s_l is the log-variance. Unlike the modality-specific latent Gaussians in UMFNet, this distribution is supervised toward a clean/reference depth representation under artificial depth corruption. A confidence map is computed using the structural evidence and the predicted uncertainty, and masked by depth validity. The confidence regulates the use of denoised depth features during subsequent fusion. Importantly, a high-frequency RGB edge is not assumed to be a geometrically valid depth boundary; the conditioning function must learn which cues are consistent.

For a clean depth target d_ref and valid-reference set Omega, an auxiliary heteroscedastic Gaussian negative log-likelihood is used:

L_unc = (1/|Omega|) sum_{p in Omega} 0.5*[exp(-s_p)*(mu_p-d_ref,p)^2 + s_p].

This loss validates simulated-noise reconstruction, not necessarily calibrated physical sensor uncertainty. Claims of sensor-noise handling require separate tests on realistic depth degradations or sensor-specific data.

## 3.3 Difficulty-Aware Progressive Fusion

At two intermediate encoder stages, a router predicts three sample-wise probabilities from the RGB features, depth features, local depth confidence, and structural disagreement. The alternatives are (i) RGB-only bypass, (ii) lightweight residual depth injection, and (iii) enhanced local depth-guided fusion. Differentiable soft mixing is used in training, whereas a discrete route is executed at inference to skip unselected branches. Stage 1 uses a compact confidence-gated depth injection, and Stage 4 remains RGB-centric. Relative route costs can be regularized during training; however, actual GFLOPs, latency, and route distribution must be measured independently. As the training and inference routing regimes differ, the performance impact of the hard decision must be evaluated.

## 3.4 Loss and Evaluation Protocol

The optimization combines multi-output saliency loss, boundary loss, optional depth Gaussian NLL, and optional routing-cost regularization. The model is selected using an independent validation split; final test sets are not used in checkpoint selection. The performance protocol reports S-measure, E-measure, weighted F-measure, and MAE with identical postprocessing across all compared models. Repeat-run summary statistics (mean ± standard deviation) may only be reported if independently repeated seeds are actually available.

## Scientific attribution

Wang, M., Shi, F., Cheng, X., Jia, C., and Chen, S., "Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection," CVPR, 2026, pp. 41489–41498. The cited prior work studies cross-modal latent Gaussian alignment and confidence-guided global modulation for RGB-T. The present proposed modifications must be positioned as hypotheses tailored to RGB-D sensor degradation, with architecture differences and quantified evidence made explicit.
