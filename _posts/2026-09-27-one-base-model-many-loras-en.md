---
layout: post
title: "One Base Model, Many LoRAs: My Adventures with vLLM and Qwen"
date: 2026-09-27 00:35:00 +0800
categories: [LLM, vLLM, LoRA]
lang: en
---

[阅读中文版]({% post_url 2026-09-27-one-base-model-many-loras-zh %})

> This is an engineering retrospective from 2023–2024. The code and architecture described here reflect the vLLM version available at the time. Please do not treat it as a current tutorial and copy everything into production—unless you also want a refresher course in incident debugging.

## It Started with the Classic “We Want Everything” Requirement

In 2023, large language models began finding their way into several of our core products. We quickly learned that a general-purpose model could chat, summarize, and occasionally hallucinate with impressive confidence—but making it truly understand a business domain required more than an ever-growing prompt.

Production workflows contain plenty of stable, recurring knowledge: domain rules, task logic, output formats, and common processing patterns. Repeating all of that in every prompt is like reintroducing your company’s org chart to a coworker every morning. Technically possible, but nobody wants to live that way.

Different teams therefore began post-training models so those recurring capabilities could live in the parameters rather than in a very long preamble. Reality soon handed us a GPU bill:

- full fine-tuning for every business was expensive;
- storing a complete model for every business produced enormous artifacts;
- deploying a separate base model for every business consumed memory at an alarming rate.

LoRA offered a much more civilized arrangement: keep the base model frozen, and train only a small set of low-rank update weights for each business.

That eased the pressure on the training side. Then inference raised its hand and asked, “What about me?”

If every LoRA adapter still required a complete copy of the base model at serving time, we would simply spend during deployment what we had saved during training. What we really wanted was the usual engineering wish list:

- load the base model once, or as close to once as possible;
- serve multiple business-specific LoRAs from one inference service;
- select the right adapter for each request;
- allow requests using different adapters to share continuous batches;
- save memory without letting one business accidentally borrow another business’s behavior.

That was why I started exploring Multi-LoRA in vLLM. Not because the name sounded fashionable, but because GPUs are expensive.

## The LoRA Equation Is Short; the Engineering Story Is Not

For an ordinary linear layer, the original computation is:

$$
y = Wx
$$

With LoRA, the base weight $W$ remains unchanged and a low-rank update is added:

$$
y = Wx + sBAx
$$

$A$ projects the input into a lower-dimensional space, $B$ maps it back, and $s$ is a scaling factor. The equation looks friendly enough to create the dangerous impression that the implementation is merely “two extra matrix multiplications.”

Once a service hosts multiple adapters and different requests in the same batch can select different ones, the equation becomes:

$$
y_i = Wx_i + s_{k_i}B_{k_i}A_{k_i}x_i
$$

Only one subscript, $k_i$, has been added. The engineering complexity, however, has discovered exponential growth. Behind that little subscript are several questions:

- How are adapter weights loaded and managed?
- Which adapter belongs to each request?
- After continuous batching, which adapter belongs to each token?
- How should $A$ and $B$ be sharded under tensor parallelism?
- How do different model architectures and module names map into the framework?
- If an adapter introduces extra tokens, who owns the vocabulary boundaries?

LoRA’s mathematics fits on a sticky note. A production Multi-LoRA design quickly fills a whiteboard.

## Round One: Building Multi-LoRA by Hand

At the time, the official solution was not yet mature, while the business could not simply sit around and wait for the framework to grow up. The first phase was therefore about implementing the complete serving path ourselves.

### Stack the A and B Matrices First

A single adapter contains one pair of $A$ and $B$ matrices. To keep multiple adapters resident, I stacked their weights along an adapter dimension:

```text
lora_A: [num_loras, rank, input_dim]
lora_B: [num_loras, output_dim, rank]
```

The base layer still computes $Wx$. Then, using the adapter index assigned to each token, the system selects the correct $A_k$ and $B_k$, computes the update, and adds it to the output:

```python
base_output = base_linear(x)
lora_output = dispatch_and_apply_lora(
    x,
    lora_a_stacked,
    lora_b_stacked,
    adapter_indices,
)
output = base_output + lora_output
```

At this point, everything looked pleasantly straightforward. In engineering stories, that is usually the cue for the main conflict to begin.

### The Hard Part Was Routing, Not Multiplication

vLLM uses continuous batching. A physical batch can contain tokens from several requests: business A may use adapter 1, business B may use adapter 2, and another request may insist on using the untouched base model.

We therefore cannot apply one pair of LoRA weights to the entire batch. The scheduler must preserve the relationship between requests and adapters, expand it into a token-level mapping, and pass that mapping into the LoRA kernel.

The most troublesome bugs here are often the ones that do not crash. If the mapping shifts, a token may quietly use somebody else’s adapter. The text can remain fluent, the logs can look perfectly peaceful, and the business behavior can still be wrong.

It is a bit like a food delivery arriving on time in a sealed bag, only for you to discover that your salad has been replaced by your neighbor’s fried chicken. Every operational metric says “success”; the semantics disagree.

That was when it became clear to me that Multi-LoRA was not a small feature inside a model layer. It was an end-to-end systems problem spanning request handling, scheduling, batching, model execution, and kernels.

### Tensor Parallelism: Linear Layers Have Personalities Too

Inside vLLM, not every linear layer behaves like a plain `torch.nn.Linear`. Under tensor parallelism, we must distinguish at least:

- Column Parallel Linear;
- Row Parallel Linear;
- QKV Packed Linear;
- Merged Column Parallel Linear.

Each layer type shards weights along different dimensions, so the sharding and aggregation points for LoRA’s $A$ and $B$ matrices also differ.

For example, Q, K, and V projections are often packed into one linear operation inside the model, while the adapter checkpoint may still store `q_proj`, `k_proj`, and `v_proj` separately. The loader must recognize that all three belong to one packed layer and preserve their ordering, widths, and tensor-parallel partitions.

The MLP side has the same kind of problem. A model may compute gate and up projections together while the adapter stores them as independent modules. An explicit packed-module mapping is required; we cannot expect them to recognize each other by destiny at runtime.

### From “It Runs” to “The Business Is Actually Using It”

The first implementation did not remain a proof of concept. It was deployed into the inference service and began handling real business LoRA requests.

That distinction mattered. A prototype only needs to answer, “Can it run?” A production system has to answer several more questions:

- Does every request consistently reach the correct adapter?
- Are businesses isolated while sharing the same base model?
- Can the service remain stable with limited inference resources?
- Can a new business be added without cloning another complete model service?

The deployment preserved at inference time the resource advantage LoRA had already provided during training. It also demonstrated that sharing a base model and routing adapters per request was viable in a real business path.

Once a system reaches production, however, it begins honestly listing all of its personality traits:

- LoRA ranks and slot counts were hard-coded;
- loading, activation, eviction, and unloading had to be maintained manually;
- model code and LoRA logic were tightly coupled;
- Embedding, LM Head, and Sampler did not share one coherent design;
- every upstream vLLM change turned migration into an open-book exam in which the textbook had also changed editions.

None of this made the first phase a failure. On the contrary, only a system serving real traffic can turn abstract “maintenance risk” into a concrete list of engineering problems.

## Round Two: Adopting Official Multi-LoRA and Teaching It About Qwen

As vLLM’s official Multi-LoRA framework matured, the goal shifted from maintaining the entire solution ourselves to integrating Qwen correctly with the common infrastructure.

The official architecture separated responsibilities more clearly:

- `LoRAConfig` managed maximum rank, adapter count, extended vocabulary, and related settings;
- the LoRA Model Manager handled adapter registration, activation, removal, and slots;
- LoRA Layer Wrappers supported different tensor-parallel layer types;
- Mapping routed requests or tokens to the correct adapters;
- each model described only its own structural differences and module mappings.

This was a welcome change. General infrastructure problems belonged in the framework; a model adapter should not secretly moonlight as an entire serving platform.

In January 2024, I completed the Qwen LoRA integration, which became the `Add qwen-lora support` change. The code was shorter than the first implementation, but there were still plenty of semantic boundaries to verify.

## A Few “Small” Problems in the Qwen Integration

When an engineer calls something a “small problem,” it usually means it has not yet become large enough to require a postmortem.

### 1. `supports_lora` Is Only an Entry Ticket

The framework first needs to know whether a model supports LoRA, so Qwen must explicitly declare that capability and accept a `LoRAConfig` during construction.

But adding a flag does not complete the integration. LoRA configuration can affect model initialization, especially the sizes of Embedding and LM Head. The real work has only begun.

### 2. Everyone Has a Different Name for the Same Thing

A common MLP mapping in the official framework is:

```text
gate_proj + up_proj -> gate_up_proj
```

The Qwen implementation at the time used:

```text
w2 + w1 -> gate_up_proj
```

When names do not match, adapter weights cannot find their destination layers. The integration therefore had to translate Qwen’s module names into packed-module relationships understood by the framework.

Qwen also had names such as `c_proj` and `c_attn`. Simply adding a suffix to a target list was unsafe because the same name could appear in different parts of the model, with different underlying types, dimensions, and wrapping support.

### 3. A Matching Name Does Not Mean the Layer Fits the Wrapper

The Model Manager usually identifies a target module by name and then replaces the original layer with a LoRA wrapper. Name matching is only the first gate; the object must also be a supported layer type.

If a module matches a naming rule but cannot be converted into `BaseLayerWithLoRA`, registering the original layer with the Manager breaks later assumptions. Weight loading and mapping updates will try to access LoRA attributes that do not exist.

A safer boundary looks like this:

```python
new_module = from_layer(module)
if new_module is None:
    continue
register_lora_module(new_module)
```

An unsupported layer should be skipped with a record or rejected explicitly. Letting an ordinary layer pretend to support LoRA is the software equivalent of sneaking into a group chat and hoping nobody asks who invited you.

### 4. Extending the Vocabulary Is More Than Stretching the Embedding

Some adapters introduce new tokens. It is tempting to think that increasing the input Embedding size is sufficient.

It is not. Vocabulary boundaries affect at least three parts of a generative model:

1. the input Embedding;
2. the output LM Head;
3. the valid sampling range used by the Sampler.

The base vocabulary, each adapter’s extended vocabulary, and padding introduced for kernel alignment are also three different concepts.

Suppose the base vocabulary size is $V$, each adapter can introduce up to $E$ extra tokens, and the service holds $N$ adapters. The physical Embedding capacity may need to account for:

$$
V + N \times E
$$

That does not mean one request should be able to sample every adapter’s extra tokens. The Sampler must know the valid range for the active adapter. Otherwise, business A may generate a token owned by business B—the delivery mix-up has now spread from the meal to the menu.

Embedding, LM Head, and Sampler therefore have to be designed as one system.

### 5. Packed LoRA Fears the Bug Where Every Shape Is Right and the Order Is Wrong

Packed layers reduce kernel launches and improve execution efficiency, but they impose strict ordering constraints.

Suppose the model stores projections in the order `w2, w1`, while the adapter loader writes them as `w1, w2`. Every tensor shape may still match, and the program may run without complaint, but the two LoRA updates will be applied to the wrong output slices.

This kind of bug is an excellent teacher: matching shapes prove only that a tensor fits, not that it belongs there. Correctness also requires module semantics, weight order, and execution slices to agree.

## What Changed Between the Two Phases?

| Dimension | Hand-Built Phase | Official Multi-LoRA Phase |
|---|---|---|
| Primary goal | Unblock and launch the business path | Integrate Qwen with a common framework |
| Weight management | Maintain stacked A/B weights manually | Managed by the Model Manager |
| Request routing | Maintain token indices ourselves | Use the framework’s Mapping |
| Model coupling | LoRA logic embedded deeply in model code | Model mainly declares capabilities and mappings |
| Tensor parallelism | Handle each layer type manually | Reuse official Layer Wrappers |
| Extended vocabulary | Incomplete support | Coordinate Embedding, LM Head, and Sampler |
| Lifecycle | Loading and unloading are our responsibility | Managed consistently by the framework |
| Long-term maintenance | Solves the immediate business problem | Easier to evolve with upstream vLLM |

The second phase required less code not because the problems disappeared, but because common problems had moved into the right abstraction layer.

## Lessons That Stayed with Me

### The Simpler the Equation, the Easier It Is to Underestimate Its Social Life

`BAx` itself is not difficult. The difficulty comes from making it coexist with continuous batching, dynamic scheduling, tensor parallelism, model variation, weight lifecycles, and vocabulary management.

Once a feature enters a production system, the hard part is often not the feature itself but its relationships with everything already there.

### Walking the Low-Level Path Makes High-Level Abstractions Legible

The hand-built implementation was not the final architecture, but it forced me to work directly with A/B stacking, token routing, and parallel-layer sharding. Later, when reading the official design, its abstractions corresponded to concrete problems rather than being merely a collection of class names.

Falling into a hole does not automatically make anyone wiser. Drawing it on the map does.

### Model Integration Is Much More Than Flipping a Switch

A complete integration must answer:

- How do adapter module names map to model modules?
- How are weights combined for packed layers?
- Which layer types can be replaced by LoRA wrappers?
- Do input and output vocabularies need extension?
- Does the Sampler understand the valid vocabulary range?
- How are dimensions partitioned under tensor parallelism?

`supports_lora = True` is an introduction, not a diploma.

### Keep Unproven Paths Explicit

In Multi-LoRA, silent compatibility is often more dangerous than an immediate error. Unknown module types, unexplained adapter weights, and inconsistent packed mappings should be explicitly skipped, recorded, or rejected.

An exception at least asks someone to investigate. A wrong result may quietly walk into production wearing a perfectly normal output as a disguise.

### Historical Code Records Decisions, Not Just Answers

Looking back at the early implementation, the hard-coded values, temporary logging, and unfinished experiments are easy to spot. They also preserve how the problems were discovered and why the design changed.

A technical retrospective that shows only the final elegant architecture is like a hiking guide containing nothing but a summit photo: the view is beautiful, but it says nothing about how many stairs were involved.

## Closing Thoughts

This exploration began with a straightforward goal: load multiple Qwen LoRA adapters into one vLLM service, share a base model, and avoid giving every business its own complete set of GPU resources.

In the first phase, I built Multi-LoRA from A/B weight stacking, token routing, and tensor-parallel layer handling, then deployed it into a real business path. In the second phase, I moved to the official architecture and focused on Qwen’s module mapping, layer wrapping, and extended-vocabulary support.

If I had to summarize the experience in one sentence, it would be this:

> The hard part of Multi-LoRA is not performing two more matrix multiplications. It is making sure the right adapter reaches the right token and the right model layer, at the right time and under the right parallel strategy—without wandering onto the wrong set.

The equation is only the beginning of the story. Whether it survives production depends on the long chain of systems details that follows.

---

**Version note:** This article describes engineering work performed in 2023–2024 against the vLLM and Qwen implementations available at the time. vLLM’s LoRA architecture has continued to evolve, and current APIs, module organization, and support may differ. Treat this as a retrospective on Multi-LoRA system design and model integration, not as a current usage guide.
