# QIPC: Quantum-Inspired Protein Sequence Collapse

A sandbox for exploring **Wave-Function-Collapse-style algorithms** applied to protein sequence generation under biochemical constraints.

**Not a predictor.** A thought experiment: *Can local constraints + probabilistic collapse produce biophysically plausible sequences?*

### Quick Start
\`\`\`bash
pip install -r requirements.txt
python examples/basic_generation.py
\`\`\`

### Features
- **Hydrophobic core modeling** — constrains hydrophobic AAs toward sequence center
- **Hard motifs** — enforce CxxC zinc fingers, disulfide patterns, etc.
- **Forbidden patterns** — exclude PPP runs, charge clusters, restriction sites
- **Secondary structure hints** — soft preferences (α-helix, β-sheet, coil)
- **Codon optimization** — DNA synthesis with restriction site avoidance
- **Backtracking WFC loop** — Monte Carlo collapse with constraint propagation

### Example
\`\`\`python
config = QIPCConfig(
    aa_constraints=AminoAcidConstraints(
        hard_motifs=["CxxC"],
        forbidden_patterns=["PPP", "DDD"],
        target_charge_range=(-5, 5),
    ),
    gen=GeneratorConfig(length=100, temperature=0.7),
)
gen = QIPCGenerator(config)
protein = gen.generate_protein_sequence()
dna = gen.generate_dna_for_sequence(protein)
\`\`\`

### What This Is & Isn't
- Exploration of constraint-based generative models  
- Sandbox for testing WFC variants  
- Not validated for actual protein design  
- Not a replacement for FoldX, Rosetta, or ESM-2  
