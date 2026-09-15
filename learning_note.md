# Learning Notes — transmodel

Last updated: 2026-09-15 21:09:01 +08:00 (Asia/Hong_Kong)

These Q&A notes summarize the discussion of `transmodel_learning_guide.pdf`, including Lesson 02, “Define the prediction contract,” Lesson 03 examples, the form of a world model, future language exploration, and the training process in Lesson 09. The timestamp records this revision; update it whenever these notes are revised.

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
