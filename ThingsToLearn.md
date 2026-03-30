Things You Should Learn / Research
Biomechanics fundamentals (most directly useful):

Wolff's Law and the Mechanostat — Frost 2003 is the key paper (already in your references). Understand the concept of a "lazy zone" where bone neither forms nor resorbs. This is the core physics of what your simulation models.
Strain Energy Density (SED) — the scalar driving force in Huiskes remodeling. Understand why SED = sigma^2/(2E) and how it differs from stress or strain alone.
Cortical vs cancellous bone — why E differs by 10-15x and how load paths change with a cortical shell.
FEA concepts (to understand your solver output):

Hexahedral element formulation (C3D8) — what Jacobian determinants mean, why negative Jacobians = inverted elements, what aspect ratio does to accuracy.
Mesh convergence — why doubling element count and checking if results change is the gold standard for FEA credibility.
Boundary conditions and their artifacts — stress concentrations at fixed nodes are artificial; understanding this helps you interpret results.
Clinical context (to frame the story for a PI):

Maxillary expansion mechanics — how RPE (Rapid Palatal Expansion) devices work; your model is essentially a biological version of this.
Functional matrix hypothesis (Moss) — soft tissue function shapes skeletal form. Your tongue-posture hypothesis is a specific instance.
Craniofacial growth and development — McNamara or Proffit's textbooks. Suture biology in adolescents is fundamentally different from adults (patent vs fused).
Visualization / presentation:

PyVista (Python VTK wrapper) — if you want genuinely impressive 3D bone renderings, this is the tool. Matplotlib 3D is inherently limited.
ParaView — free, open-source, industry-standard FEA visualization. Can read CalculiX .frd files directly.
Want me to start working through Tier 1? I'd suggest tackling items 1-4 as a batch — they're all independent and I can parallelize them with subagents. The result would be a dramatically better-looking animation with physically meaningful remodeling dynamics.

