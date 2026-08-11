## Documentation

Treat code as the single source of truth. By default, do not add natural-language explanations, including documentation, doc comments, and code comments.

Keep this default strict because explanations are easy to add and therefore proliferate, create duplicate ownership with the code, resist verification, and mislead decisions when wrong or stale.

### Decision Process

Before writing an explanation, apply these gates in order:

1. **Code gate:** Can ordinary code reading reveal the information without a large investigation? If yes, do not write it. Continue only when the information cannot be learned from the code or rediscovering it would require a large investigation every time.
2. **Future-value gate:** Will recording the information materially improve your execution of future tasks? If no, do not write it. Continue only when the information will save useful future work.
3. **Clause-value gate:** State the useful current information directly. Then test every additional clause independently:
   - Remove the clause.
   - Would its absence permit a concrete, plausible wrong action or decision, or require material rediscovery in a future task?
   - If no, delete it. If yes, keep only the minimum wording that prevents that cost.

   Apply this test to prior states and rejected alternatives alike. Prefer `B` to both `A used to be true, but now B` and `not A, but B`. Keep the `A` clause only when it passes the test independently. Every retained clause is another claim to verify and keep synchronized with the code.

### Rewriting Existing Explanations

Do not patch an isolated sentence or clause. Reconsider the entire explanation against the gates above, draft its complete replacement, and replace it as a unit. This prevents ad hoc additions, redundancy, and obsolete information from accumulating.

## Prompting

### Build the Runtime Prompt

For each model call:

1. Define the model's exact runtime task and list the context it will actually receive.
2. Apply this inclusion gate to every candidate context item: if the model can satisfy the runtime task's stated requirements without it, omit it. Pass only the minimum required context. Omit application-wide design, implementation history, change history, and information needed only by the developer unless a specific part passes this gate. What you need while implementing the call differs from what the model needs when the call runs.
3. Audit every reference such as "given X," "provided X," or "the X above." Verify that X exists in the model's actual runtime context. If it does not, add X only when it passes the inclusion gate; otherwise rewrite the instruction to remove the reference. Never rely on information known only during implementation.

### Constrain Parsed Outputs

When application code parses the model's output:

1. Inspect the model API, SDK, and provider capabilities for structured outputs, constrained decoding, schemas, grammars, or an equivalent mechanism.
2. Use an available compatible constraint. Do not rely on free-form output and best-effort parsing when the output can be enforced.
3. Determine whether the schema or grammar is actually passed to the model as runtime context, independently of what the decoder enforces. Use wording such as "given the schema" only when the model can read that schema. If the model does not receive it, do not claim that it does.
4. Always explain the required output format in the pr

## Testing

Treat tests as debt, a rate-limiting constraint, a compromise, and the last resort. They add authoring, execution, and maintenance costs while offering only probabilistic protection.

Before writing, keeping, or recommending a test, apply these gates in order:

1. **Static-guarantee gate:** Can the property be guaranteed statically? If yes, use that guarantee and do not use a test for it. Continue only if the property cannot be guaranteed statically.
2. **Future-value gate:** Is the property worth protecting in the future? If no, do not test it. Continue only if future bugs affecting the property should be easier to detect.
3. **Purpose gate:** Is this test being added merely because something was implemented, so that a passing result can be cited as evidence that the implementation is correct? If yes, do not write it; passing examples do not prove current correctness. Write or keep a test only when its expected behavior is justified independently of the current implementation and it targets a concrete, plausible future bug or regression that it would make easier to detect.

Only write or retain a dynamic test when the candidate passes all three gates.
