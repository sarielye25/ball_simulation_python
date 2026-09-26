# Learning Notes — transmodel

Last updated: 2026-09-26 +08:00 (Asia/Hong_Kong)

These Q&A notes summarize the discussion of `transmodel_learning_guide.pdf`, `neuron_network_supplementary.pdf`, and `loss_function_supplementary.pdf`, including the prediction contract, examples, world models, future language exploration, neural-network layers and activations, supervised training, and loss-function terminology. The timestamp records this revision; update it whenever these notes are revised.

## Q: What is my current goal for this project?

**A:** Train a learned world model named **transmodel** to predict the cuboid's motion over a specified time interval: displacement during the interval and velocity at its end. Here, `d` means signed displacement, not total distance travelled or absolute position.

For now, finish studying the PDF and leave the existing implementation unchanged. Start with fixed physical conditions, establish accurate predictions through evaluation, and then introduce additional varying conditions gradually.

## Q: Does reducing the number of input variables improve prediction accuracy? Aren't three or four inputs easy for a GPU?

**A:** Three or four inputs are computationally tiny. Fewer inputs do not automatically mean better accuracy. Starting with fixed conditions mainly simplifies data collection, evaluation, and diagnosis of mistakes.

The important distinction is between fixing a physical condition and hiding a condition that varies:

| Experiment | Meaning for the model |
|---|---|
| Mass stays fixed and is omitted from the inputs | The model learns motion for that particular mass. |
| Mass varies and is provided as an input | The model can learn how motion depends on mass, with adequate training coverage. |
| Mass varies but is hidden | Identical visible inputs may require different predictions. |
| Mass is supplied but has only one value throughout training | The data do not teach how changing mass affects motion. |

Fixing mass or friction does not remove its physical effect. The model learns a relationship specific to those conditions. More varied conditions require adequate examples of their effects and combinations; a faster GPU cannot replace missing information or training coverage.

## Q: Why can a missing variable make prediction inaccurate?

**A:** A missing variable matters when it varies between cases, changes the required prediction enough to matter, and cannot be inferred from the available inputs or history.

For a cuboid already sliding positively without stopping during the interval:

\[
a = \frac{F}{m} - \mu_k g.
\]

With force 10 N, kinetic friction coefficient 0.2, and gravity 9.8 m/s², acceleration is 3.04 m/s² for a 2 kg cuboid and 0.54 m/s² for a 4 kg cuboid. The same current velocity and force therefore produce different future velocities. A predictor that receives neither mass nor information sufficient to infer its effect cannot uniquely distinguish those cases.

The effect of an omission is not always large. Its importance depends on the operating range and required accuracy.

## Q: Why is target position unnecessary when current position and velocity matter?

**A:** Current position is where the cuboid is now. Next position is where physics takes it. Target position is where we want it to go.

A controller uses the goal and current state to choose a force. The world model predicts motion under the supplied force. If two experiments have identical initial states, physical conditions, and force schedules, changing the desired destination does not change their motion.

“Under a supplied force” means that the action throughout the prediction interval is specified. If a controller changes the force during that interval in response to a target, the force schedule can differ and so can the motion. The first experiment avoids this ambiguity by holding force constant throughout the interval.

## Q: Does current position have to be an input to transmodel?

**A:** On a uniform floor with no boundaries or position-dependent effects, absolute position does not affect displacement or velocity change. A learned function can predict those changes from velocity and force under fixed physical conditions. Current position is then added to predicted displacement to obtain absolute next position.

Moving an entire experiment 100 m to the right should add 100 m to its predicted next position while leaving displacement and velocity unchanged. This assumption fails if position changes the physics, such as near walls or across different surfaces.

## Q: How does the proposed world model differ from the current physics calculation?

**A:** `cuboid_translation_x/physics.py` calculates total displacement from a constant positive push starting at rest, followed by coasting until rest. `simulation.py` uses that displacement to update the stopping position.

Transmodel's proposed task is to predict displacement and velocity after a specified interval from a possibly moving initial state. The cuboid need not have stopped when that interval ends. The existing stopping calculation is a useful analytical case, but does not provide arbitrary fixed-interval transitions.

Simple cases can already be calculated quickly using mechanics. A neural network has not yet demonstrated a speed or accuracy advantage here. The learning goal is to train and evaluate a predictor against a physics reference we understand.

## Q: Should an industrial robot's world model include as many variables as possible?

**A:** It should contain enough relevant information to meet its prediction requirements within its intended operating conditions. Maximizing the input count is not the objective.

Variable mass and surface friction may matter substantially. Paint colour usually does not affect this mechanical prediction. Air resistance may be negligible at the chosen speeds and tolerances. Some unmeasured conditions can be estimated from motion history rather than supplied directly.

Useful inputs can improve prediction, while irrelevant or noisy inputs can introduce misleading correlations and additional data requirements. Ask: “Could this missing information change the answer enough to matter for this task?” A 1 cm stopping tolerance and a 10 μm tolerance can require different models of the same object.

## Q: Is including every possible variable an industrial standard?

**A:** No universal industrial standard prescribes that approach or requires a learned world model. Requirements and applicable standards depend on the robot and application.

A common engineering approach is to define operating conditions and acceptable errors, identify influential factors, measure or estimate them where appropriate, account for uncertainty, and validate representative and difficult cases.

For example, a robot may receive a known payload mass, estimate its effect from measurements, or use control that tolerates a specified mass range. Industrial systems inevitably face unmeasured influences such as wear, disturbances, and sensor errors.

Industrial readiness requires more than prediction accuracy: timing, control stability, fault handling, and behavior outside supported conditions also matter. Agreement with simulator-generated labels alone does not establish accuracy on physical hardware.

## Q: What is the equivalent issue in an LLM?

**A:** The closest analogy is whether the available context contains enough information to determine a useful answer. Asking for the acceleration of a block under 10 N without specifying mass and friction does not determine one numerical answer.

An LLM can explain the dependencies, ask for information, or state assumptions. General training cannot reliably reveal the mass of a particular block that has not been described. Additional context or tools can supply missing facts, just as motion history can help estimate hidden conditions in a dynamics problem.

More context is not automatically better: irrelevant or contradictory information can make the task harder. A typical LLM predicts tokens, whereas transmodel is intended to predict physical quantities. Fluent text alone does not establish reliable physical prediction.

## Q: What is an “unclearly defined physical setup”? Does it only mean hidden variables?

**A:** Hidden varying conditions are one possibility, but ambiguity can also concern the prediction task itself:

- Unspecified horizon: predict velocity after 0.1 s, after 1 s, or after the push ends?
- Unspecified action schedule: hold 10 N throughout the interval or release it halfway?
- Ambiguous output: does `d` mean signed displacement, distance travelled, or absolute position?
- Unspecified mechanics: can the cuboid tip or encounter a wall?
- Unrecorded changing conditions: does mass or friction change between experiments?

A hidden variable does not necessarily make the problem unclear. “Mass is unknown, lies between 1 and 3 kg, and recent motion history is supplied to help estimate its effect” is a clear problem involving incomplete information.

## Q: What is a clear first prediction contract for transmodel?

**A:** Given current velocity and a force held constant throughout a specified interval, predict signed displacement during that interval and velocity at its end, for a fixed mass on a horizontal uniform surface with fixed friction and gravity, constrained to one-dimensional motion without tipping or impacts.

Before collecting training data, specify the interval, numerical physical conditions, allowed velocity and force ranges, and resting/stopping rules. Evaluate displacement and velocity errors separately in physical units, and check repeated predictions as well as one-step predictions.

## Session added: 2026-09-15 21:09:01 +08:00 (Asia/Hong_Kong)

Sariel is studying Lesson 03. The following Q&As capture this conversation, including clarifications and the decision to retain language understanding as a crucial future direction. They supplement the earlier notes without changing implementation.

## Q: Is a world model a formula containing all necessary variables and parameters? What is transmodel's actual form?

**A:** A world model's role is to predict how an environment evolves, including the effects of actions. Different mathematical structures can serve that role. It need not describe the entire physical world; transmodel's world is the constrained cuboid experiment.

For the current prediction contract:

\[
(\widehat d,\widehat v_{\text{next}})=f_\theta(v,F).
\]

The inputs are current velocity and applied force; the outputs are predicted signed displacement and final velocity. Hats indicate predictions. Mass, friction, gravity, and the prediction interval stay fixed for the first experiment, but their effects remain present.

The chosen structure with adjustable parameters defines a **family of possible functions**. One particular set of parameter values selects one specific function. Training searches for useful parameter values within that family.

Physical parameters, such as mass and friction coefficients, differ from neural-network parameters, which are weights and biases. A network can approximate motion without individual weights corresponding to named physical quantities or discovering readable Newtonian equations.

## Q: What does the proposed neural network look like?

**A:** Lesson 06 proposes a small starting candidate: two numerical inputs, one hidden layer with 16 ReLU neurons, and two linear outputs. It has 82 trainable parameters: 32 input-to-hidden weights, 16 hidden biases, 32 hidden-to-output weights, and 2 output biases.

Each hidden neuron forms a weighted combination of the inputs, adds a bias, and applies a nonlinear activation. The output layer combines the hidden responses. Training adjusts the weights and biases. This is a candidate to evaluate, not an established optimum.

Lesson 06 proposes displacement and **velocity change** as outputs, while our current notes use displacement and **final velocity**. Both are valid, related by \(v_{\text{next}}=v+\Delta v\). Keep the chosen convention consistent in labels, training, and prediction; these notes use final velocity.

## Q: Why does the guide ask me to describe the simulation in human language? Does the network read that description?

**A:** The description is a specification for the person designing the experiment. Statements such as “the force stays constant throughout 0.1 seconds” and “mass and friction are fixed” determine simulator settings, model inputs, and reference labels.

The proposed numerical network does not read these sentences. Their meaning is implemented through the experiment and data. Writing a clear description exposes ambiguity before it becomes inconsistent training data.

## Q: How do I obtain an untrained world model and integrate it into this project?

**A:** Create the chosen network using a machine-learning library such as PyTorch. The library supplies layer building blocks and standard initializations; no pretrained model download is necessary for this small predictor. Initially, its weights have no learned knowledge of the cuboid's dynamics.

The verified physics reference generates input–answer pairs. Training feeds inputs to the network, compares predictions with reference answers, and updates the weights. After evaluation on unseen examples, a prediction process can load the learned weights and use the network to estimate future motion. It must use the same architecture, input/output conventions, and scaling as training.

At the time of this discussion, the project has no trained transmodel. The existing push-and-coast stopping calculation does not yet supply arbitrary transitions over the chosen fixed interval; those reference labels must be made trustworthy before training.

## Q: If transmodel is a numerical function rather than an LLM, is it still a world model? Can world models understand human language?

**A:** Yes, transmodel is a narrow learned world model. “Numerical prediction function” describes how it works; “world model” describes its role. Language understanding is not required for that role.

A world model can use human language when the system is designed and trained to do so. For example, a language component can turn an instruction into a representation used alongside scene or state information to predict what happens next. Understanding an instruction and predicting its physical consequences are distinct capabilities that a system can combine.

Language can leave physical details unspecified: “push left” does not determine force magnitude or duration. A language-capable system still needs enough context, explicit conventions, or clarification to define the action whose effects it predicts.

## Q: Can language understanding remain a crucial future exploration direction?

**A:** Yes. Sariel explicitly chose language understanding as a crucial future vertical exploration direction, motivated partly by curiosity and the appeal of the capability. This decision was recorded in README.md during this conversation.

The first numerical predictor remains useful as a foundation. Later experiments can explore connecting a trained language component to it or training a predictor that directly uses a language representation alongside physical information. The architecture is undecided. An example challenge is interpreting “give the cuboid a gentle push” in a defined physical context. Evaluate both instruction interpretation and resulting motion predictions, including unfamiliar wording.

## Q: Does choosing examples mean covering as many different circumstances as possible? Are those categories the supervised-learning labels?

**A:** Aim for useful coverage of the intended operating range and important boundaries, rather than simply maximizing example count. An example is an input paired with its reference output; its **label** is that output:

\[
\underbrace{(v,F)}_{\text{input}}
\longrightarrow
\underbrace{(d,v_{\text{next}})}_{\text{label}}.
\]

“Resting,” “sliding,” and “stopping” are useful categories for organizing and checking coverage. They are not the outputs of our current predictor. Predicting continuous motion quantities is supervised **regression**, not classification.

For illustrative fixed conditions \(m=2\) kg, \(\mu_s=0.4\), \(\mu_k=0.2\), \(g=9.8\) m/s², and \(\Delta t=0.1\) s, with force held throughout the interval:

| Initial velocity | Applied force | Behavior | Reference label: (displacement, final velocity) |
|---|---|---|---|
| 0 m/s | 5 N | Remains at rest | (0 m, 0 m/s) |
| 0 m/s | 10 N | Starts moving | (0.0152 m, 0.304 m/s) |
| 0.5 m/s | 0 N | Slows but remains moving | (0.0402 m, 0.304 m/s) |
| 0.1 m/s | 0 N | Stops inside the interval | (approximately 0.00255 m, 0 m/s) |

These are analytical examples under the stated assumptions, not trained predictions.

## Q: Can I classify cases by force magnitude into no motion, stopping inside the interval, and stopping afterward?

**A:** First distinguish a resting cuboid from an already-moving one. Force magnitude alone is insufficient: a force below the maximum static friction does not imply that an already-moving cuboid is stationary. The last two examples above have identical force but different outcomes because their initial velocities differ.

Vary velocity and force together. Cover rest, starting, sliding in both directions, coasting, braking, stopping, and cases near regime boundaries. Static friction governs whether a resting body remains at rest; kinetic friction opposes current sliding motion.

“Still moving when the interval ends” does not necessarily mean “will stop afterward.” That depends on the future action schedule. In the guide's positive 10 N example, maintaining that force keeps accelerating the body. The model's label concerns the specified interval, not an unspecified eventual stopping event.

When stopping occurs inside an interval, resolve the stopping instant and apply the rest rule for the remaining time. Continuing the previous friction acceleration past zero velocity can produce an unphysical reversal. A sufficiently strong applied force can cause a real reversal, but friction alone cannot.

## Q: Below the static-friction limit, should the model learn that friction equals the applied force?

**A:** For a body at rest on the horizontal floor, with the applied force as the only other horizontal force, ideal static friction balances it within the limit:

\[
f_s=-F,\qquad |F|\leq\mu_smg.
\]

Equal magnitude and opposite direction produce zero net force. Maximum static friction is a limit, not a constant friction value. In the example, the limit is 7.84 N, but a 5 N push at rest is balanced by -5 N of static friction.

Transmodel's labels teach the resulting behavior: zero displacement and zero final velocity for these resting examples. Friction is not currently an output. Correct motion predictions do not establish that the network internally represents the explicit equation “friction equals minus applied force.” The physics reference uses force balance to calculate the answer; transmodel learns to predict the answer from examples.

## Q: If the model is a family of functions, who performs the automatic correction during supervised learning?

**A:** The training process performs repeated numerical adjustments automatically. The model itself calculates predictions with its current parameters; the surrounding training components perform different jobs:

| Component | Responsibility |
|---|---|
| Model | Calculate predictions from inputs and current parameters |
| Loss function | Measure disagreement with reference labels |
| Automatic differentiation and backpropagation | Calculate derivatives of loss with respect to the parameters |
| Optimizer | Use gradients to update parameters |
| Training loop | Repeat these operations over batches of examples |

For one update, supply a batch, calculate predictions and loss, calculate parameter gradients, and let the optimizer update the weights. The next batch uses the updated parameters. Sariel configures the process and evaluates its results; he does not manually correct every prediction or adjust every weight.

## Q: How does gradient descent know which parameters to change?

**A:** The loss value measures error. Its gradient describes how a small change to each parameter would locally change that loss, with other parameters held fixed. A positive derivative means a small increase in the parameter locally increases loss; a negative derivative means it locally decreases loss.

Ordinary gradient descent updates parameters according to:

\[
\theta_{\text{new}}=\theta_{\text{old}}-\eta\nabla_\theta L.
\]

Here, \(\eta\) is the learning rate, controlling update size. Backpropagation efficiently calculates derivatives through the network using the chain rule. PyTorch can automate those calculations. Backpropagation calculates gradients; the optimizer uses them to change parameters. Adam is a common optimizer that uses gradients and information from previous updates.

## Q: Can I see one automatic correction numerically?

**A:** Consider a dimensionless toy model \(\widehat y=wu\), with input \(u=2\), label \(y=3\), and initial weight \(w=1\). Its prediction is 2, and its squared-error loss is:

\[
L=(wu-y)^2=1.
\]

The derivative is:

\[
\frac{\partial L}{\partial w}=2(wu-y)u=-4.
\]

With learning rate 0.1, the optimizer calculates:

\[
w_{\text{new}}=1-0.1(-4)=1.4.
\]

The new prediction is 2.8 and the loss is 0.04. Nobody manually selected 1.4 as the correction. It follows from the example, model, loss, derivative, and learning rate. A neural network applies the same principle to many connected parameters.

## Q: How are numerous updates organized? What do I still need to do myself?

**A:** A **batch** is a group of examples used for one update. An **epoch** is one pass through the training dataset. A **training run** usually includes many epochs and parameter updates. For example, 10,000 examples in batches of 100 give 100 updates per epoch; 50 epochs give 5,000 updates in one run.

The human work is defining the prediction task, providing trustworthy labels, choosing the training setup, and checking unseen cases. The software performs the repeated adjustments.

Automatic adjustment is not automatic discovery of correct physics. One update may improve average batch loss while worsening some examples, and a large update may even increase loss overall. Bad labels can teach wrong behavior, and low training error does not guarantee good predictions on unseen cases.

## Q: Does the model keep correcting itself when I use it after training?

**A:** Ordinarily, no. During prediction, its weights remain fixed and it only performs the forward calculation. Further learning requires explicitly running an update process again with a suitable source of feedback. Generating a prediction alone supplies no reference answer for supervised correction.

## Q: So do I manually write a training-loop script? Does the PDF cover it?

**A:** Yes. Sariel writes a script that coordinates prediction, loss calculation, gradient calculation, and parameter updates. PyTorch supplies differentiation and optimizer operations; he does not need to implement all their underlying mathematics himself.

Lesson 09, **“How iteration changes the weights” (page 9)**, explains a gradient-descent update, backpropagation versus optimization, batches, epochs, and learning rate. Its section **“Where training lives in your Python project”** describes how reference data, the trainable model, and the training process fit together.

The PDF covers the training process conceptually, without implementation code or a complete script-writing walkthrough. Lesson 11 points to PyTorch learning resources for the later implementation stage. Sariel will continue studying the guide from Lesson 03; the discussion of training does not require jumping ahead to implementation now.

## Session added: 2026-09-17 10:32:13 +08:00 (Asia/Hong_Kong)

These notes capture the follow-up discussion of Lesson 05 and the supplementary PDF on 16–17 September. Sariel is clarifying hidden layers, bias updates, and ReLU. He asked to set aside the full page 3 walkthrough until the first two topics are clear; the later ReLU question is recorded below without resuming that deferred walkthrough. Do not assume these topics are mastered merely because explanations have been provided.

## Q: What is a hidden layer? Is it the part that performs calculations?

**A:** A layer is a stage of computation within one prediction. In this feedforward network, a hidden layer contains several neurons that calculate intermediate values between the supplied inputs and final outputs. The output layer also performs calculations; computation is not exclusive to hidden layers.

The example has three input values, four hidden neurons, and two outputs. Each hidden neuron receives all three inputs, applies its own weights and bias, and then applies an activation function. Each output combines all four hidden values using its own weights and bias.

“Hidden” means intermediate, not inaccessible. We can inspect its calculated values, such as:

\[
h=(0.7,0,0.6,0).
\]

We supply reference labels for the final motion predictions, but normally no “correct hidden-neuron value.” Hidden values are recalculated for each input. They are not additional measurements or trainable parameters.

## Q: Why have four neurons when all receive the same inputs? Do they calculate four target values?

**A:** The four hidden neurons calculate four intermediate responses, not four targets. The example still has only two prediction targets: future position and future velocity. The neurons receive the same inputs but use different weights and biases, so they can respond differently.

For an illustrative pair receiving scaled velocity and force:

\[
h_1=\operatorname{ReLU}(z_v+z_F),\qquad
h_2=\operatorname{ReLU}(z_v-z_F).
\]

At \(z_v=1\) and \(z_F=1\), their outputs are 2 and 0. These are toy calculations, not physical laws.

The output layer combines different responses to represent relationships that one response cannot express. This flexibility is useful when motion behaves differently around rest, positive and negative sliding, and stopping. Training adjusts the neurons together according to their combined prediction error. We do not normally assign one neuron to “friction” and another to “stopping”; their learned roles need not have simple physical meanings.

Four neurons are an illustrative architecture choice, not a requirement. More neurons increase representational capacity but do not automatically improve predictions. If neurons performed identical calculations, they would not provide the distinct responses this explanation relies on.

| Quantity | What determines its count? |
|---|---|
| Input values | The information supplied to the model |
| Hidden neurons | An architecture choice |
| Output values | The quantities to predict |

## Q: Where does a neuron's bias come from? Is it initially random and then adjusted using the loss?

**A:** A bias needs an initial value, but it does not have to be random. Common choices include zero or small random values, depending on the layer and initialization method. Libraries provide defaults. The supplement's bias of -0.1 was deliberately chosen for easy arithmetic; it was neither trained nor measured from the physical system.

During a prediction, the bias stays fixed. During training, the loss measures final prediction error, backpropagation calculates its derivative with respect to the bias, and the optimizer updates the bias. For ordinary gradient descent:

\[
b_{\text{new}}=b_{\text{old}}-\eta\frac{\partial L}{\partial b}.
\]

Biases and weights are both trainable parameters. After training, ordinary prediction uses their learned values without changing them.

## Q: Do we use the same loss function to adjust biases and weights? Why?

**A:** Yes. In our setup, one overall loss measures errors in the final motion predictions. Both weights and biases affect those predictions, so the same objective provides feedback to both. Each parameter has its own derivative, and therefore can receive a different update.

For a simple model and squared-error loss:

\[
\widehat y=wu+b,\qquad L=(\widehat y-y)^2,
\]

the derivatives are:

\[
\frac{\partial L}{\partial w}=2(\widehat y-y)u,
\qquad
\frac{\partial L}{\partial b}=2(\widehat y-y).
\]

Changing the weight changes its contribution in proportion to the input \(u\). Changing the bias adds an offset directly. This explains why the updates differ even though they use the same loss. In a full network, backpropagation follows the connected calculations to obtain each parameter's derivative; hidden neurons do not need separate labels or separate losses.

## Q: Why does ReLU keep positive values and replace negative values with zero? Is it because the cuboid can only move forward or remain still?

**A:** No. ReLU is a chosen mathematical activation, defined by:

\[
h=\max(0,q).
\]

Here, \(q\) is an internal weighted sum, not the cuboid's velocity or displacement. The rule introduces a nonlinear bend: a neuron is inactive in one input region and responds in another. Combining such responses allows richer functions than a single global affine mapping. ReLU is one activation choice, not a physical law or the only way to introduce nonlinearity.

If \(q=-0.5\), its ReLU output is zero. This says that one neuron contributes zero for this input; it does not say the cuboid cannot move backward or that the cuboid is at rest.

Nonnegative hidden values can still produce negative final predictions because output weights and biases can be negative. For a toy output calculation:

\[
\widehat v_{\text{next}}=-2h_1+0.5h_2,
\]

hidden values \(h_1=1\) and \(h_2=0\) produce a negative output of -2, interpreted according to the chosen output scale.

Our proposed network uses ReLU in the hidden layer and unrestricted linear outputs. Putting ReLU directly on the final velocity output would forbid negative velocity predictions, which would be inappropriate when backward motion is allowed. Any forward-only behavior in the current positive-push experiment comes from its initial conditions and allowed forces, not from the reason for choosing ReLU.

## Session added: 2026-09-17 16:56:57 +08:00 (Asia/Hong_Kong)

Sariel is studying page 2 of `loss_function_supplementary.pdf`. This discussion exposed a terminology problem: saying squared loss “amplifies larger errors” can sound as though the gradient or weight adjustment must also be quadratic. Sariel understood the calculation; the unclear meaning of “amplification” caused the confusion. Keep loss values, gradients, and update sizes distinct when explaining this topic.

## Q: What does “The loss tells you how your chosen objective penalizes the errors” mean?

**A:** The **chosen objective** is the mathematical goal selected for training (选定的优化目标), such as minimizing average squared prediction error. To **penalize an error** means to assign it a numerical cost (给错误赋予数值上的代价), not to literally punish the model.

In plain English: **“The loss measures how bad the model's mistakes are, according to the scoring rule you chose.”** Absolute error and squared error are different scoring rules, so they can prefer different compromises between mistakes.

## Q: Does squaring the error require compensation when adjusting weights?

**A:** No separate compensation step is needed. Let the signed error be \(e=\widehat y-y\). For squared loss and ordinary gradient descent:

\[
L=e^2,\qquad
\frac{\partial L}{\partial\widehat y}=2e,\qquad
\frac{\partial L}{\partial w}=2e\frac{\partial\widehat y}{\partial w},\qquad
w_{\text{new}}=w_{\text{old}}-\eta\frac{\partial L}{\partial w}.
\]

The **loss value** scores the mistake. The **gradient** describes the local slope. The **learning rate** \(\eta\) (eta, 学习率) scales the parameter update. Training does not directly subtract the loss value from a weight.

Differentiation gives the slope of the chosen loss; it does not undo the square or compensate for an unwanted effect. Using \(\tfrac12 e^2\) merely removes the factor of 2 from the derivative. The learning rate does not need to increase when the error increases: with the same learning rate, a larger gradient already produces a larger ordinary gradient-descent step.

## Q: If the gradient is only at “e-level,” what exactly is amplified?

**A:** Sariel's informal “e-level” means **linear in the error**, rather than quadratic. That correctly describes the gradient of squared loss with respect to the prediction. The word “amplification” must identify which quantity is being compared:

| Quantity | When error magnitude becomes ten times larger |
|---|---|
| Squared-loss value, \(e^2\) | Becomes 100 times larger |
| Prediction-gradient magnitude, \(2|e|\) | Becomes ten times larger |
| Individual weight-update magnitude in ordinary gradient descent | Becomes ten times larger if the learning rate and prediction sensitivity to that weight are the same |

**Squared loss does not make the gradient quadratic.** If “amplification” means a tenfold error creates a more-than-tenfold prediction gradient, squared loss does not provide that behavior. Earlier wording that suggested otherwise was ambiguous.

The useful comparison is with absolute loss. For nonzero errors, \(L=|e|\) has a prediction gradient of +1 or -1: its magnitude is constant. Squared loss has prediction gradient \(2e\): its magnitude grows with error size. Thus larger errors exert a stronger corrective push than smaller errors under squared loss, whereas absolute loss gives them equal push magnitudes at the prediction level. At zero, absolute loss has a corner and no ordinary derivative.

Squaring also does not always increase a numerical value: \(0.2^2=0.04\). The precise claim is that squared loss gives larger errors disproportionately greater **cost relative to smaller errors**.

## Q: Does a linear gradient mean squaring did not change the gradient?

**A:** No. “Linear” does not mean “unchanged.” The choice of loss determines its gradient:

| Loss | Gradient with respect to prediction | Consequence |
|---|---|---|
| Signed error, \(e\) | \(1\) | Minimizing it keeps pushing the prediction downward, even when it is already too low; opposite signed errors can cancel in a sum |
| Absolute error, \(|e|\) | \(\operatorname{sign}(e)\), for nonzero error | Correct direction, constant push magnitude |
| Squared error, \(e^2\) | \(2e\) | Correct direction, push magnitude proportional to error |

If we invented the rule “apply a corrective push proportional to prediction error,” we would rediscover the prediction gradient of squared loss, up to a constant factor. The loss states the mathematical goal served by that rule. Actual weight gradients also include how the weight affects the prediction; contributions from multiple examples can reinforce or cancel one another.

## Q: Does choosing squared loss change the learned answer, or only the training steps?

**A:** It can change the learned answer because it changes the preferred compromise when a model cannot satisfy every example.

Suppose a model must output one shared prediction for three equally weighted targets: **0, 0, and 9**.

| Shared prediction | Total squared loss | Total absolute loss |
|---|---|---|
| 0 | \(0+0+81=81\) | \(0+0+9=9\) |
| 3 | \(9+9+36=54\) | \(3+3+6=12\) |

Squared loss is minimized at **3, the mean**. Absolute loss is minimized at **0, the median**. Using averages instead of totals leaves these minimizers unchanged.

Under squared loss, reducing the large error on the target 9 is worth introducing smaller errors on the two zeros. At prediction 3, the prediction-gradient contributions are \(6,6,-12\), which sum to zero. Under absolute loss, each nonzero error supplies a push of equal magnitude, so the two zero targets favor a different compromise.

**Key terminology takeaway:** “Squared loss emphasizes large errors” means quadratic costs and an error-dependent corrective push. It does not mean squared gradients, an automatically increased learning rate, or a separate compensation step. This choice can change what the model learns, not just how far it moves during training.

## Session added: 2026-09-17 17:32:10 +08:00 (Asia/Hong_Kong)

Follow-up discussion of scales, weighting the components of one example's loss, and averaging losses across a batch (including the formula Sariel encountered on page 5). The distinction between changing how an error is measured and changing the actual physical error remains important.

## Q: What does a scale do, and does it enter the gradient?

**A:** Dividing an error by a reference scale with the same units makes the ratio dimensionless. Sariel connected this with dimensional reasoning from equilibrium constants in chemistry. For a position scale of \(s=0.01\,\text{m}\), an error of 1 cm becomes a normalized error of 1, and an error of 5 cm becomes 5.

**Scaling expresses an error relative to a chosen reference size; it does not itself change the prediction or its physical error.** Updating the model's parameters can change the actual error.

For a fixed positive scale and signed error \(e=\widehat y-y\):

\[
\ell=\left(\frac{e}{s}\right)^2,\qquad
\frac{\partial\ell}{\partial\widehat y}=\frac{2e}{s^2},\qquad
\frac{\partial\ell}{\partial w}=\frac{2e}{s^2}\frac{\partial\widehat y}{\partial w}.
\]

The scale enters through the chain rule: differentiating the square gives \(2e/s\), and differentiating the normalized prediction with respect to the original prediction supplies another factor of \(1/s\).

If the numerical scale is 0.01, the normalized error is 100 times the original numerical error, and the gradient with respect to the original prediction is 10,000 times that of unscaled squared error. This numerical comparison holds the original units and model parameterization fixed. Differentiating with respect to the normalized prediction \(\widehat z=\widehat y/s\) instead gives \(2e/s\); these are derivatives with respect to different coordinates.

No automatic compensation cancels this effect. For a single loss under ordinary gradient descent, multiplying the learning rate by \(s^2\) would cancel the uniform gradient scaling. With several loss components scaled differently, one learning-rate adjustment generally cannot cancel their changed relative influences.

## Q: How can I choose the scale instead of selecting one arbitrarily?

**A:** Sariel connected the scale on page 3 of `loss_function_supplementary.pdf` with data normalization. One systematic choice is to use each variable's standard deviation in the training data. For a value \(x\), training mean \(\mu\), and training standard deviation \(\sigma\), standardization is:

\[
z=\frac{x-\mu}{\sigma}.
\]

If both the prediction and reference target are standardized with the same \(\mu\) and \(\sigma\), their difference is:

\[
\widehat z-z
=\frac{\widehat y-\mu}{\sigma}-\frac{y-\mu}{\sigma}
=\frac{\widehat y-y}{\sigma}.
\]

The mean cancels, so calculating loss between standardized predictions and targets is equivalent to dividing the physical error by the target's training-set standard deviation. This gives a concrete, data-derived scale: an error of 1 means an error equal to one training standard deviation.

Fit the mean and standard deviation on the training split only, then reuse those fixed values for validation, test data, and inverse conversion back to physical units. This prevents information from validation or test data from influencing the training setup.

Standard deviation is a useful starting choice because it puts variables with different units and typical spreads onto comparable numerical scales. It is not automatically an acceptance tolerance or a statement of physical importance. If one standard deviation of displacement is much larger than the maximum error acceptable for the application, the statistical scale and the acceptance tolerance answer different questions.

## Q: Are scale and importance the same thing?

**A:** They have different interpretations, but both affect the loss and gradient. For one example, write \(\ell\) (pronounced “ell”) as:

\[
\ell=
\lambda_d\left(\frac{\widehat d-d}{s_d}\right)^2+
\lambda_v\left(\frac{\widehat v-v}{s_v}\right)^2.
\]

Here, \(d\) is signed displacement, \(v\) is final velocity, the positive scales express reference error sizes in the corresponding units, and the nonnegative coefficients \(\lambda_d,\lambda_v\) express additional priorities. The effective coefficients on the original squared errors are \(\lambda_d/s_d^2\) and \(\lambda_v/s_v^2\).

Thus, saying “scale does not describe importance” is useful as a distinction in purpose, but should not imply that scale has no effect on the training trade-off. A smaller scale increases that component's influence, all else equal.

## Q: What does manually weighting velocity loss more strongly actually do?

**A:** It multiplies velocity's contribution to the parameter gradients. Define \(\ell_d\) and \(\ell_v\) as the two normalized squared errors. For any trainable parameter \(w\):

\[
\frac{\partial\ell}{\partial w}
=\lambda_d\frac{\partial\ell_d}{\partial w}
+\lambda_v\frac{\partial\ell_v}{\partial w}.
\]

The **loss weights** \(\lambda_d,\lambda_v\) are priorities chosen for training; they are distinct from the neural network's trainable weights.

Suppose a shared parameter has a distance-loss gradient of +2 and a velocity-loss gradient of -3. Distance favors decreasing the parameter; velocity favors increasing it.

| Loss weights | Combined gradient | Update with learning rate 0.01 |
|---|---|---|
| Both 1 | \(2-3=-1\) | \(+0.01\) |
| Distance 1, velocity 2 | \(2+2(-3)=-4\) | \(+0.04\) |

Increasing velocity's coefficient strengthens its contribution. This is especially relevant when improving one output worsens another: the loss specifies the preferred compromise. Doubling a coefficient does not guarantee halving the resulting error or making that component dominate, because the underlying gradients also matter.

## Q: Why average the losses within a batch? Is it like observing many examples to see a tendency?

**A:** There are two separate reasons: **combining examples improves the evidence for an update; dividing by their count normalizes its numerical scale.** For a batch of \(B\) examples:

\[
L=\frac{1}{B}\sum_{i=1}^{B}\ell_i,\qquad
\nabla_w L=\frac{1}{B}\sum_{i=1}^{B}\nabla_w\ell_i.
\]

One example's gradient may help that example while hurting others. Combining representative examples can give a less noisy estimate of the gradient across the data. Sariel related this to his reading of Taleb: individual observations vary, while many observations can reveal a tendency. This is a useful intuition, not a guarantee or a precise attribution of a statistical theorem to Taleb.

Averaging does not eliminate variation or guarantee the true tendency. Biased or strongly correlated samples, and rare extreme observations, can still make an average misleading. The familiar reduction in sampling noise relies on assumptions about the samples and their variability; more observations alone do not resolve every problem.

## Q: Why divide by batch size instead of simply summing?

**A:** We want batch size to control how many examples inform an update without automatically multiplying the update size.

Suppose three examples give gradients \(2,4,6\) for one parameter, evaluated at the same current model weights. Duplicate every example within the batch, giving \(2,4,6,2,4,6\). This repeats the same evidence without adding information.

| Quantity | Three entries | Six entries, each original duplicated |
|---|---|---|
| Sum of gradients | 12 | 24 |
| Average gradient | 4 | 4 |
| Update using the sum, learning rate 0.01 | -0.12 | -0.24 |
| Update using the average, learning rate 0.01 | -0.04 | -0.04 |

The sum doubles the step merely because the batch has twice as many entries. The average leaves it unchanged. This example compares a single update from each batch, not two sequential updates that recalculate gradients after changing the weights.

For a fixed batch, the sum and average have the same gradient direction; averaging divides the magnitude by \(B\). In ordinary gradient descent, averaging is equivalent to using the sum with a learning rate divided by batch size:

\[
\eta\left(\frac{1}{B}\sum_i g_i\right)
=\frac{\eta}{B}\left(\sum_i g_i\right),
\]

where \(g_i\) is an example's gradient contribution.

**Combining examples addresses dependence on a single observation. Averaging rather than summing removes the automatic factor of batch size from the gradient.** It makes “how many examples we consult” and “how big a step we take” easier to control separately. New examples can still change the average gradient, and changing batch size can change training behavior; averaging does not guarantee identical training runs or remove every need to tune the learning rate.

## Session added: 2026-09-17 17:52:03 +08:00 (Asia/Hong_Kong)

Discussion of Huber loss on page 7 of `loss_function_supplementary.pdf`. Sariel's central takeaway is that Huber reduces the tendency of examples with very large errors to dominate batch loss and training. The precise qualification is that it limits their influence rather than guaranteeing that they can never dominate.

## Q: What does “threshold” mean in Huber loss?

**A:** A **threshold** (阈值／分界值) is a boundary value where a rule changes. Huber's positive threshold \(\delta\), pronounced “delta,” determines where the loss switches from quadratic to linear growth.

Compare the error's magnitude \(|e|\) with the threshold, not its signed value. In this PDF, \(e=(\widehat y-y)/s\) is already a normalized, dimensionless error. With a position scale of 0.01 m and \(\delta=1\), the switch occurs at a physical error magnitude of 1 cm. Both positive and negative errors use the same magnitude rule.

The threshold does not divide acceptable errors from ignored errors. Both sides contribute to the loss.

## Q: What are the Huber rules, explicitly?

**A:** First calculate the normalized error, then check its magnitude:

\[
H_\delta(e)=
\begin{cases}
\frac12 e^2, & |e|\leq\delta,\\[4pt]
\delta\left(|e|-\frac12\delta\right), & |e|>\delta.
\end{cases}
\]

- **At or below the threshold:** use half the squared error.
- **Above the threshold:** use the linear rule in error magnitude.

For \(\delta=1\), the outer rule simplifies to \(|e|-0.5\):

| Normalized error | Rule | Loss | Gradient with respect to normalized error |
|---|---|---|---|
| 0.5 | Quadratic | 0.125 | 0.5 |
| 1 | At threshold | 0.5 | 1 |
| 2 | Linear | 1.5 | 1 |
| -2 | Linear | 1.5 | -1 |
| 4 | Linear | 3.5 | 1 |

The derivative is:

\[
H'_\delta(e)=
\begin{cases}
e, & |e|\leq\delta,\\[4pt]
\delta\,\operatorname{sign}(e), & |e|>\delta.
\end{cases}
\]

Within the threshold, the corrective push grows with error. Beyond it, the derivative's magnitude stays at \(\delta\), while its sign still indicates the correction direction. The loss itself continues increasing without a fixed upper limit.

The outer formula makes the pieces meet smoothly: at \(e=\delta\), both loss formulas give \(\frac12\delta^2\), and both slopes give \(\delta\). The negative boundary also joins smoothly. A smaller threshold starts limiting the derivative sooner; a larger threshold keeps more errors in the quadratic region.

## Q: How does Huber prevent large errors from dominating a batch?

**A:** More precisely, **Huber reduces the disproportionate influence of very large errors: their loss grows linearly instead of quadratically, and their derivative with respect to normalized error stops growing.**

Keep the notation distinct: \(\ell_i\) is one example's loss, potentially combining displacement and velocity terms; \(L=\frac1B\sum_i\ell_i\) is the batch-average loss. Huber is applied to each error component before combining and averaging. It does not clip the batch average afterward.

For a simple batch with one error component per example, compare normalized errors of 1 and 100:

| Penalty | Loss at error 1 | Loss at error 100 | Derivatives at errors 1 and 100 |
|---|---|---|---|
| Half squared error, \(\frac12e^2\) | 0.5 | 5,000 | 1 and 100 |
| Huber with \(\delta=1\) | 0.5 | 99.5 | 1 and 1 |

Half squared error makes the larger error contribute 10,000 times the loss and 100 times the prediction-error derivative. Huber reduces that loss ratio to 199 and gives equal derivative magnitudes here. Averaging across the batch does not change these relative contributions.

**Important precision:** An extreme error can still dominate the numerical batch loss because Huber's linear tail is unbounded. Huber also does not impose an absolute bound on weight gradients. For one component:

\[
\frac{\partial H_\delta(e)}{\partial w}
=H'_\delta(e)\frac1s\frac{\partial\widehat y}{\partial w}.
\]

The derivative with respect to normalized error is bounded, but scales, loss coefficients, and the prediction's sensitivity to a weight still affect each example's contribution to the parameter update.

This can help with occasional corrupted labels or noisy measurements. For this simulation, first investigate unusually large errors: a rare stopping event may be important physics to learn, rather than bad data to suppress.

## Session added: 2026-09-23 +08:00 (Asia/Hong_Kong)

## Q: Why should training cases remain in batch training after they currently pass the error tolerance?

**A:** All cases share the same model parameters. An update based on one batch can improve its cases while making a previously accurate case inaccurate again. Therefore, the baseline will keep every training case eligible for shuffled mini-batches instead of permanently removing cases that currently pass. Each update will use the batch's average differentiable loss; tolerance checks belong to evaluation rather than selecting which cases may train the model.
