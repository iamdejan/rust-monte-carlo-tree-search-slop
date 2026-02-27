# Monte-Carlo Tree Search (Slop)

This is an example of Monte-Carlo Tree Search implementation. However, this is generated with Kilo Code's agent using free models, such as:
- Minimax 2.5 (to generate the logic); and
- Kimi 2.5 (to generate Rustdoc and code comments)

This is a demo project to test:
1. the capabilities of Kilo Code and its free models; and
2. whether I can understand agent-generated code or not.

## Did it work? What are the results?

1. I think Minimax 2.5 can get you to 80% of the goal. However, you need to be able to independently verify the results by validating yourself the generated unit tests.
2. For someone who has never studied MCTS before, the code can be hard to read, especially when the prompt specifically asks the agent to implement based on The University of Queensland's [Mastering Reinforcement Learning](https://uq.pressbooks.pub/mastering-reinforcement-learning/) but the agent did not follow the pseudocode. I don't know whether the agent can look up to the link I provided or not, but comparing the code with the pseudocode given by UQ, it's clear that the code does not strictly follow that resource, which hampered my learning curve. I had to ask Kimi 2.5 to generate the comments.

## Why is it called "slop"?

Well, the thing is, I don't know this algorithm, so I don't know the correctness. This is different with A\* algorithm, which I understand well. Looking briefly at the generated unit tests, I think the generated code works as intended, but I am not willing to assume that. Therefore, I will still call this "slop."

## Prerequisites

Before you run the code, install [Pixi](https://pixi.prefix.dev/latest/) by following [their guide](https://pixi.prefix.dev/latest/installation/).

## How to run this code

If you know Pixi well, you know that all of the commands can be found in [pixi.toml](./pixi.toml).
- Build: `pixi run build`
- Lint: `pixi run lint`
- Fix lint (for simple linter errors): `pixi run lint-fix`
- Run the code: `pixi run start`
- Generate HTML from Rustdoc: `pixi run doc`
- Generate HTML from Rustdoc, then open: `pixi run doc-open`
