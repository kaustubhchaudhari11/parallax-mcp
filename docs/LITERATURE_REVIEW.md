# PARALLAX Literature Review

**Historical note, 25 October 2026.** This review was written on
21 October 2026. It compares neighboring work. It does not report
PARALLAX measurements, and it does not claim novelty. Figures below
are the cited authors' results.

**Status:** 21 October 2026.
This review records sources that were opened and what they support.
It does not claim that PARALLAX is novel. Several systems already replay
agent trajectories, inject faults or prompts into tool channels, or test
agents with metamorphic relations. The last section states the
differences the prototype was being built to maintain. Those differences
are design choices until an evaluation shows they matter.

## How sources were checked

Primary pages opened for this review:

- arXiv or publisher pages for the papers cited in the next sections
- the Model Context Protocol specification revision `2025-06-18`
- the installed MCP Python SDK 2.3.0, for implementation facts rather
  than academic claims

A source that appeared only as a citation inside another paper, or only
as a search snippet, is listed at the end and is not used as evidence.

## Metamorphic testing

Chen, Cheung, and Yiu introduced metamorphic testing as a way to generate
follow-up test cases from relations among inputs when a direct oracle for
every output is hard to obtain (Technical Report HKUST-CS98-01,
Department of Computer Science, Hong Kong University of Science and
Technology, 1998;
<https://cse.hkust.edu.hk/faculty/scc/publ/CS98-01-metamorphictesting.pdf>).

Cho, Ruberto, and Terragni collected 191 metamorphic relations for
natural-language tasks, implemented 36 of them, and report about 560,000
tests on three language models (arXiv:2511.02108, submitted 3 November
2025; ICSME 2025, pp. 174–186, DOI 10.1109/ICSME64153.2025.00025). The
relations apply to NLP tasks. They do not call MCP tools.

Zheng, Zhou, Xu, Ren, and Chen survey 93 studies on metamorphic testing
and language models (arXiv:2605.13898, 12 May 2026). Their taxonomy
includes "MT for LLMs" applied to autonomous agents, and a separate
direction in which models themselves help generate tests. The survey is
evidence that agent behavior is already inside the metamorphic-testing
literature. It does not evaluate this repository.

de Zarzà, de Curtò, Cabot, Manzoni, and Calafate apply eight
semantic-preserving transformations to 19 multi-step reasoning problems
and call the stability of the answers semantic invariance
(arXiv:2603.13173, submitted 13 March 2026, revised 16 March 2026). The
transformations include identity, paraphrase, fact reordering, expansion,
contraction, a change of context, and a contrastive formulation. They
report that Qwen3-30B-A3B produced invariant responses on 79.6% of their
items, with semantic similarity 0.91, and that larger models in their set
were less stable. Those transformations edit the problem statement. They
do not edit an MCP tool result, and the paper also studies
semantic-preserving edits only.

PARALLAX uses the metamorphic idea in a narrower place. An invariance
intervention is a follow-up case whose oracle label is required to stay
the same. A sensitivity intervention is a follow-up case whose oracle
label is required to change in a specified way. The relation is checked
against a deterministic policy, because the fixture facts are known to
the evaluator. The agent still does not see the label. This is stricter
than a pure oracle-absent metamorphic test, and it is also narrower: the
"relation" is only as meaningful as the policy in
`RESEARCH_PROTOCOL.md`.

## Indirect prompt injection and agent benchmarks

Greshake, Abdelnabi, Mishra, Endres, Holz, and Fritz showed that
instructions placed in data an application is likely to retrieve can
control an LLM-integrated application, including which APIs it calls
(AISec 2023, 26 November 2023, DOI 10.1145/3605764.3623985;
arXiv:2302.12173). They call this indirect prompt injection. H3 is a
descendant of that threat, restricted to synthetic MCP tool results and
scored against a fixed policy.

Zhan, Liang, Ying, and Kang built InjecAgent, 1,054 cases over 17 user
tools and 62 attacker tools (Findings of ACL 2024, DOI
10.18653/v1/2024.findings-acl.624; arXiv:2403.02691). Attack goals are
direct harm or data exfiltration. They report that a ReAct-prompted
GPT-4 was vulnerable on 24% of attacks, and that a reinforced hacking
prompt nearly doubled that rate. Success means the agent carries out the
attacker's instruction. PARALLAX's H3 success criterion is the complement
of a narrower event: the decision, the `action` field, and the tool name
stay inside the policy. InjecAgent does not pair that check with an
irrelevant-edit condition or a decision-relevant fact change.

Debenedetti, Zhang, Balunovic, Beurer-Kellner, Fischer, and Tramèr
published AgentDojo at NeurIPS 2024 (proceedings dated 10 December 2024;
arXiv:2406.13352). AgentDojo is an extensible environment with 97 tasks
and 629 security test cases. Agents call tools over untrusted data, and
the paper evaluates attacks and defenses together. Two methodological
points in that paper constrain PARALLAX:

- The authors argue that an LLM-simulated tool is a poor place to study
  prompt injection, because the injection can fool the simulator.
- They also avoid an LLM judge for the same reason: a successful
  injection can hijack the judge.

PARALLAX follows both points. Tool results come from a deterministic
fixture server. The score comes from a rule over those fixtures.
AgentDojo's measured outcome is whether an attacker goal succeeds while
the user task still has utility. PARALLAX's additional comparisons are
the invariance and sensitivity pairs, which are not attacker goals.

Ruan, Dong, Wang, Pitis, Zhou, Ba, Dubois, Maddison, and Hashimoto
introduced ToolEmu at ICLR 2024 ("Identifying the Risks of LM Agents with
an LM-Emulated Sandbox"). An LM emulates tools and another LM scores
risk. On their initial set of 36 toolkits and 144 cases, human review
judged 68.8% of the failures ToolEmu identified to be valid real-world
failures, and their evaluator reported failures for the safest agent
23.9% of the time. The emulation is what makes the suite broad.
PARALLAX gives up that breadth so the observation under intervention is
not itself a model sample.

## MCP security benchmarks

The Model Context Protocol specification, revision 2025-06-18, defines
MCP as a JSON-RPC 2.0 protocol among hosts, clients, and servers, used to
expose tools and context to LLM applications
(<https://modelcontextprotocol.io/specification/2025-06-18>). The
installed Python SDK is newer than that document and knows later version
identifiers, recorded in `ARCHITECTURE.md`. Protocol attacks against
commercial hosts are a different project.

Guo, Liu, Ma, Deng, Zhu, Di, Xiao, and Wen present MCPXkit
(arXiv:2508.12538, 2025). The paper organizes 31 attack methods into
direct tool injection, indirect tool injection, malicious-user attacks,
and attacks that use model-inherent behavior. It describes tool poisoning
through hidden instructions in tool descriptions, and it reports that
agents in their study relied heavily on those descriptions and had
trouble separating external data from executable instructions. MCPXkit
is an attack toolkit. Its question is whether the attacks work.
PARALLAX does not poison tool descriptions, does not ship exploit
tooling, and scores a stated incident policy instead of attack completion.

Yang, Gao, Wu, Chen, Li, and Wang present MCPSecBench
(arXiv:2508.13220, 2025). They describe 17 attack types on four surfaces
(client, protocol, server, and host) and a playground that runs those
attacks against MCP hosts. The abstract reports that every surface
compromised at least one of Claude, OpenAI, and Cursor in their
evaluation, and that the protection mechanisms they tried succeeded less
than 30% of the time on average. PARALLAX does not reproduce that
playground and does not use those percentages as expectations. The
relevant contrast is the target: MCPSecBench exercises host and protocol
compromise; PARALLAX edits one observation inside an otherwise honest
local session.

## Causal replay of agent trajectories

Shah, "Causal Agent Replay: Counterfactual Attribution for LLM-Agent
Failures" (arXiv:2606.08275, 2026), is the closest system in method.
The paper models a trajectory as a sequence of state, action, observation,
and an outcome score. It applies `do` interventions and re-executes the
suffix under the same stochastic policy. The implemented operations are
`do_resample`, `do_action`, `do_observation`, `do_context`, and
`do_policy`. Because the policy is stochastic, each intervention yields a
distribution, reported with intervals. On synthetic structural causal
models with planted ground truth, the paper reports that a contrastive
estimator recovered the pivotal step, and that a Shapley estimator
recovered a two-step interaction (`φ0 = 0.44`, `φ1 = 0.45`, `φ2 ≈ 0`).

Shah also reports two constraints PARALLAX adopts:

- Hosted providers are not deterministic at temperature 0, so replay
  fidelity has to be measured rather than asserted.
- Rule-based outcome functions are preferable to LLM judges when the
  score needs to be trusted. Real tools with side effects are outside
  that paper's demonstrations.

The questions differ. Causal Agent Replay asks which step caused a bad
outcome, and it validates attribution against planted causal graphs.
PARALLAX asks whether a pre-declared edit to an MCP observation leaves a
correct decision unchanged, moves it to a new correct decision, or
installs an untrusted instruction. The branch-at-observation mechanic is
similar to `do_observation`. PARALLAX does not estimate Shapley values,
does not search for a causal locus, and does not treat a user-supplied
score as the outcome. The outcome is agreement with an executable policy
that was fixed before the run.

Shah cites Zhang et al., "Which Agent Causes Task Failures and When?"
(ICML 2025, arXiv:2505.00212), for a step-level attribution accuracy of
about 14% for LLM judges on Who&When. This review did not re-read Zhang
et al. The 14% figure is used only as Shah's stated reason for preferring
intervention over a judge. It is not a number PARALLAX will compare
itself against.

## What the prototype is being built to do differently

| System | Question it is built to answer | What it changes | How it scores |
| --- | --- | --- | --- |
| InjecAgent, AgentDojo | Does an indirect injection achieve an attacker goal? | Untrusted text in tool data, often with an attacker task | Attack success and, in AgentDojo, task utility |
| MCPXkit, MCPSecBench | Which MCP attack surfaces compromise hosts? | Descriptions, servers, protocol, host behavior | Attack success on MCP hosts |
| ToolEmu | Where do tool agents create high-stakes failures? | Scenarios, with tools emulated by a model | LM evaluator, checked in part by humans |
| de Zarzà et al. | Do reasoning agents stay stable under reworded problems? | The user problem, preserving its meaning | Agreement across transformations |
| Causal Agent Replay | Which step caused a bad outcome? | An action, an observation, context, or the policy, then replay | Shift in an outcome distribution |
| PARALLAX (planned) | Does this agent apply a stated incident policy under a declared MCP observation edit? | One schema-checked tool result, after a real MCP call, from a frozen prefix | Deterministic oracle on fixture facts; labels stay out of the prompt |

The combination PARALLAX is aiming at, and has not yet demonstrated:

- a real MCP client and server, not an in-prompt fake of a tool
- deterministic fixtures, so the observation is not an LM sample
- three pre-declared relations (preserve, adapt, refuse) rather than one
  attacker goal or one attribution score
- paired factual replay from the same prefix, so a mutation effect is
  separated from a fresh sample
- exact retention of both payloads
- an oracle that cannot flow back into the agent context

Any one of those pieces exists in prior work. The claim that the
combination measures something the others do not is unproven. A
literature table is not that proof.

## Sources located and not used as evidence

These identifiers came up while searching. Their abstracts or citations
were not opened far enough to support a claim in this review:

- Hou, Zhao, Wang, and Wang, "Model Context Protocol (MCP): Landscape,
  Security Threats, and Future Research Directions," arXiv:2503.23278,
  as cited by Guo et al.
- Radosevich and Halloran, "MCP Safety Audit," arXiv:2504.03767, as cited
  by Guo et al.
- Owotogbe et al., "A Taxonomy of Runtime Faults in Model Context Protocol
  Servers," arXiv:2606.05339. The search result describes an empirical
  taxonomy of server runtime faults. That subject is adjacent and is not
  the intervention PARALLAX will apply.
- Zhang et al., Who&When, arXiv:2505.00212, beyond Shah's citation.
- AgenTracer, arXiv:2509.03312. Shah describes it as replay by
  substituting a gold action. This review did not check that description
  against the paper.

No citation in this file was invented to fill a gap. If a later milestone
depends on one of the unread papers, that paper has to be opened first.
