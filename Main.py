"""
This project is purely exploratory and for fun.

It plays with the idea of using a Wave-Function-Collapse–style algorithm
to generate *plausible* protein sequences under biochemical constraints
(hydrophobicity, charge, motifs, secondary-structure hints, etc.).

The goal is not to predict real proteins, but to reflect on a question:

    Can local constraints and probabilistic collapse rules
    produce sequences that look biophysically reasonable?

This code is a sandbox for that question & nothing more, nothing less.
"""


import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple


# basic amino acid definitions

STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")

# idrofobicità semplificata - normalize value
AA_HYDROPHOBICITY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5,
    "F": 2.8, "G": -0.4, "H": -3.2, "I": 4.5,
    "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8,
    "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# simplify ph physiological
AA_CHARGE = {
    "A": 0, "C": 0, "D": -1, "E": -1,
    "F": 0, "G": 0, "H": +1, "I": 0,
    "K": +1, "L": 0, "M": 0, "N": 0,
    "P": 0, "Q": 0, "R": +1, "S": 0,
    "T": 0, "V": 0, "W": 0, "Y": 0,
}

# simplify preference for secondary structure
SS_PREFS = {
    "H": set("ALMEQKR"),       # elica
    "E": set("VILFYWT"),       # foglietto beta
    "C": set("GSPNDCTY"),      # coil / loop
}

# mapping  aa
AA_TO_CODONS = {
    "A": ["GCT", "GCC", "GCA", "GCG"],                # Alanine
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],  # Arginine
    "N": ["AAT", "AAC"],                              # Asparagine
    "D": ["GAT", "GAC"],                              # Aspartic Acid
    "C": ["TGT", "TGC"],                              # Cysteine
    "Q": ["CAA", "CAG"],                              # Glutamine
    "E": ["GAA", "GAG"],                              # Glutamic Acid
    "G": ["GGT", "GGC", "GGA", "GGG"],                # Glycine
    "H": ["CAT", "CAC"],                              # Histidine
    "I": ["ATT", "ATC", "ATA"],                       # Isoleucine
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],  # Leucine
    "K": ["AAA", "AAG"],                              # Lysine
    "M": ["ATG"],                                     # Methionine (Start)
    "F": ["TTT", "TTC"],                              # Phenylalanine
    "P": ["CCT", "CCC", "CCA", "CCG"],                # Proline
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],  # Serine
    "T": ["ACT", "ACC", "ACA", "ACG"],                # Threonine
    "W": ["TGG"],                                     # Tryptophan
    "Y": ["TAT", "TAC"],                              # Tyrosine
    "V": ["GTT", "GTC", "GTA", "GTG"],                # Valine

    # STOP CODONS
    "STOP_CODONS": ["TAA", "TAG", "TGA"],
    #START
    "START_CODONS": ["ATG"],
}


# config dataclasses

@dataclass
class AminoAcidConstraints:
    allowed_aminoacids: str = "standard20"
    hard_motifs: List[str] = field(default_factory=list)        # motivi obbligatori (es. ["CxxC"])
    forbidden_patterns: List[str] = field(default_factory=list) # pattern vietati (es. ["PPP"])
    max_run_length: int = 4                                     # max aa uguali consecutivi
    target_charge_range: Optional[Tuple[int, int]] = None       # (min, max) carica netta totale
    strict_secondary_structure: bool = False
    secondary_structure: Optional[str] = None                   # stringa tipo "HHHCCE..."

@dataclass
class CodonConstraints:
    forbidden_codons: List[str] = field(default_factory=lambda: list(STOP_CODONS))
    forbidden_sites: List[str] = field(default_factory=list)    # es. ["GAATTC", "GGATCC"]
    forbidden_patterns: List[str] = field(default_factory=list) # es. ["TATA", "AATAAA"]
    avoid_rare_codons: bool = False                             # per demo non implementiamo freqs reali
    stop_codons_allowed_only_at_end: bool = True

@dataclass
class GeneratorWeights:
    w_hydrophobic_core: float = 1.0
    w_run_penalty: float = 1.0
    w_forbidden_pattern: float = 2.0
    w_charge_penalty: float = 0.5

@dataclass
class GeneratorConfig:
    length: int = 100
    temperature: float = 1.0
    max_backtrack_steps: int = 200
    max_restarts: int = 5
    diversity_seed: Optional[int] = None   # per reproducibilità

@dataclass
class QIPCConfig:
    aa_constraints: AminoAcidConstraints = field(default_factory=AminoAcidConstraints)
    codon_constraints: CodonConstraints = field(default_factory=CodonConstraints)
    weights: GeneratorWeights = field(default_factory=GeneratorWeights)
    gen: GeneratorConfig = field(default_factory=GeneratorConfig)

#utility finction

def softmax(scores: List[float], temperature: float) -> List[float]:
    if temperature <= 0:
        # deterministic
        max_idx = scores.index(max(scores))
        probs = [0.0] * len(scores)
        probs[max_idx] = 1.0
        return probs
    scaled = [s / temperature for s in scores]
    max_s = max(scaled)
    exps = [math.exp(s - max_s) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]

def sample_with_probs(options: List[str], probs: List[float]) -> str:
    r = random.random()
    cum = 0.0
    for o, p in zip(options, probs):
        cum += p
        if r <= cum:
            return o
    return options[-1]

# qipc wfc generator

class QIPCGenerator:
    def __init__(self, config: QIPCConfig):
        self.config = config
        if self.config.gen.diversity_seed is not None:
            random.seed(self.config.gen.diversity_seed)

        # set allowed aa set
        if self.config.aa_constraints.allowed_aminoacids == "standard20":
            self.allowed_aas = set(STANDARD_AA)
        else:
            #  default standard20
            self.allowed_aas = set(STANDARD_AA)

    # --- public api ---

    def generate_protein_sequence(self) -> Optional[str]:
        """Genera una sequenza proteica con WFC e vincoli. Ritorna None se fallisce."""
        for restart in range(self.config.gen.max_restarts):
            seq = self._wfc_once()
            if seq is not None and self._check_hard_motifs(seq):
                return seq
        return None
    def _choose_codon_for_aa(self, aa: str, dna_so_far: str, is_last: bool) -> Optional[str]:
        cc = self.config.codon_constraints

        # START CODON
        if len(dna_so_far) == 0:
            return "ATG"

        # codoni
        possible = AA_TO_CODONS.get(aa, [])
        if not possible:
            return None

        # no stop codon in the center
        possible = [c for c in possible if c not in cc.forbidden_codons]

        # STOP only at the end
        if cc.stop_codons_allowed_only_at_end and not is_last:
            possible = [c for c in possible if c not in STOP_CODONS]
        else:
            possible = possible + STOP_CODONS

        if not possible:
            return None

        #choose between the possible codon
        random.shuffle(possible)
        for codon in possible:
            new_dna = dna_so_far + codon
            if self._check_dna_partial(new_dna):
                return codon

        return None


    def _check_dna_partial(self, dna: str) -> bool:
        cc = self.config.codon_constraints
        # evita siti di restrizione parziali
        for site in cc.forbidden_sites:
            if site in dna:
                return False
        # evita pattern vietati
        for pat in cc.forbidden_patterns:
            if pat in dna:
                return False
        return True


    def _check_dna_constraints(self, dna: str) -> bool:
        # qui puoi aggiungere controlli globali se vuoi
        return self._check_dna_partial(dna)

    def generate_dna_for_sequence(self, seq: str) -> Optional[str]:
        """Mappa aa -> DNA rispettando vincoli base sui codoni."""
        dna = ""
        for i, aa in enumerate(seq):
            codon = self._choose_codon_for_aa(aa, dna, is_last=(i == len(seq)-1))
            if codon is None:
                return None
            dna += codon
        if not self._check_dna_constraints(dna):
            return None
        return dna

# --- internal: wfc loop ---

    def _wfc_once(self) -> Optional[str]:
        n = self.config.gen.length
        # domini iniziali: tutti gli aa permessi per ogni posizione
        domains: List[Set[str]] = [set(self.allowed_aas) for _ in range(n)]
        sequence: List[Optional[str]] = [None] * n
        backtrack_stack = []

        steps_without_progress = 0

        while True:
            # trova posizione con min entropia > 1
            idx = self._select_position_to_collapse(domains, sequence)
            if idx is None:
                # tutte le posizioni collassate?
                if all(a is not None for a in sequence):
                    seq_str = "".join(sequence)  # type: ignore
                    if self._check_global_constraints(seq_str):
                        return seq_str
                    else:
                        return None
                else:
                    # nessuna mossa possibile -> fallimento
                    return None

            options = list(domains[idx])
            if not options:
                # dominio vuoto -> backtrack
                if not backtrack_stack or steps_without_progress > self.config.gen.max_backtrack_steps:
                    return None
                domains, sequence = backtrack_stack.pop()
                steps_without_progress += 1
                continue

            # calcola score per ogni candidate aa
            scores = [self._score_candidate(sequence, idx, aa) for aa in options]
            # vogliamo max score -> softmax
            probs = softmax(scores, self.config.gen.temperature)
            chosen = sample_with_probs(options, probs)

            # salva stato per backtracking
            snapshot_domains = [set(d) for d in domains]
            snapshot_seq = sequence[:]
            backtrack_stack.append((snapshot_domains, snapshot_seq))

            # applica scelta
            sequence[idx] = chosen
            domains[idx] = {chosen}

            # propaga vincoli a vicini
            self._propagate_constraints(domains, sequence, idx)

            steps_without_progress += 1
            if steps_without_progress > self.config.gen.max_backtrack_steps:
                return None

    def _select_position_to_collapse(self, domains: List[Set[str]],
                                     sequence: List[Optional[str]]) -> Optional[int]:
        min_entropy = None
        min_idx = None
        for i, (dom, aa) in enumerate(zip(domains, sequence)):
            if aa is not None:
                continue
            size = len(dom)
            if size == 0:
                return i
            if size == 1:
                # già determinata, saltiamo
                continue
            if (min_entropy is None) or (size < min_entropy):
                min_entropy = size
                min_idx = i
        return min_idx

    # --- scoring & propagation ---

    def _score_candidate(self, sequence: List[Optional[str]], idx: int, aa: str) -> float:
        """Score soft (più alto = meglio) per un candidato aa in posizione idx."""
        w = self.config.weights
        score = 0.0

        # 1) idrofobicità: preferisci aa idrofobici nel "core" centrale
        n = len(sequence)
        center = n / 2.0
        dist_center = abs(idx - center) / center  # 0 al centro, 1 ai bordi
        hydro = AA_HYDROPHOBICITY.get(aa, 0.0)
        # più idrofobico al centro, più idrofilo ai bordi
        score += w.w_hydrophobic_core * (hydro * (1.0 - dist_center))

        # 2) penalità per run troppo lunghi
        run_len = self._run_length_if_add(sequence, idx, aa)
        if run_len > 1:
            score -= w.w_run_penalty * (run_len - 1)

        # 3) penalità se crea pattern vietati
        seq_str = self._sequence_with_candidate(sequence, idx, aa)
        for pat in self.config.aa_constraints.forbidden_patterns:
            if pat in seq_str:
                score -= w.w_forbidden_pattern

        # 4) carica netta (soft)
        if self.config.aa_constraints.target_charge_range is not None:
            target_min, target_max = self.config.aa_constraints.target_charge_range
            charge_now = self._estimate_charge(seq_str)
            if charge_now < target_min:
                score -= w.w_charge_penalty * (target_min - charge_now)
            elif charge_now > target_max:
                score -= w.w_charge_penalty * (charge_now - target_max)

        return score

    def _propagate_constraints(self, domains: List[Set[str]],
                               sequence: List[Optional[str]], idx: int) -> None:
        """Aggiorna i domini vicini in base a vincoli semplici (es. struttura secondaria)."""
        aac = self.config.aa_constraints
        ss = aac.secondary_structure
        if aac.strict_secondary_structure and ss is not None:
            # se abbiamo info SS, restringi domain alla preferenza per ogni posizione
            for i in range(len(domains)):
                if sequence[i] is not None:
                    continue
                if i >= len(ss):
                    continue
                ss_char = ss[i]
                if ss_char in SS_PREFS:
                    allowed = SS_PREFS[ss_char] & self.allowed_aas
                    if allowed:
                        domains[i].intersection_update(allowed)

    # --- helper per scoring ---

    def _run_length_if_add(self, sequence: List[Optional[str]], idx: int, aa: str) -> int:
        """Calcola la lunghezza della run di aa uguali se metti aa in idx."""
        # scorri a sinistra e destra da idx
        run = 1
        # sinistra
        i = idx - 1
        while i >= 0 and sequence[i] == aa:
            run += 1
            i -= 1
        # destra
        i = idx + 1
        while i < len(sequence) and sequence[i] == aa:
            if i == idx:
                break
            run += 1
            i += 1
        return run

    def _sequence_with_candidate(self, sequence: List[Optional[str]], idx: int, aa: str) -> str:
        seq = sequence[:]
        seq[idx] = aa
        return "".join([x if x is not None else "X" for x in seq])

    def _estimate_charge(self, seq: str) -> int:
        return sum(AA_CHARGE.get(a, 0) for a in seq if a in AA_CHARGE)

    # --- global checks (hard) ---

    def _check_global_constraints(self, seq: str) -> bool:
        aac = self.config.aa_constraints

        # 1) max run length hard
        if aac.max_run_length is not None and aac.max_run_length > 0:
            current = 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1]:
                    current += 1
                    if current > aac.max_run_length:
                        return False
                else:
                    current = 1

        # 2) forbidden patterns hard
        for pat in aac.forbidden_patterns:
            if pat in seq:
                return False

        # 3) charge hard range (se definito come hard)
        if aac.target_charge_range is not None:
            charge = self._estimate_charge(seq)
            mn, mx = aac.target_charge_range
            if charge < mn or charge > mx:
                return False

        return True

    def _check_hard_motifs(self, seq: str) -> bool:
        """Controlla che tutti i motivi obbligatori siano presenti (hard)."""
        aac = self.config.aa_constraints
        for motif in aac.hard_motifs:
            if not self._motif_present(seq, motif):
                return False
        return True

    def _motif_present(self, seq: str, motif: str) -> bool:
        """
        Motivo tipo 'CxxC' dove 'x' = qualsiasi aa.
        Implementiamo 'x' come wildcard singola.
        """
        if "x" not in motif:
            return motif in seq

        m_len = len(motif)
        for i in range(len(seq) - m_len + 1):
            ok = True
            for j, c in enumerate(motif):
                if c == "x":
                    continue
                if seq[i + j] != c:
                    ok = False
                    break
            if ok:
                return True
        return False


# example usage

if __name__ == "__main__":
    STOP_CODONS = AA_TO_CODONS["STOP_CODONS"]

    aa_cons = AminoAcidConstraints(
        allowed_aminoacids="standard20",
        hard_motifs=["CxxC"],
        forbidden_patterns=["PPP", "DDD"],
        max_run_length=4,
        target_charge_range=(-5, 5),
        strict_secondary_structure=False,
        secondary_structure=None,
    )

    codon_cons = CodonConstraints(
        forbidden_codons=list(STOP_CODONS),
        forbidden_sites=["GAATTC", "GGATCC"],
        forbidden_patterns=["TATA", "AATAAA"],
        avoid_rare_codons=False,
        stop_codons_allowed_only_at_end=True,
    )

    weights = GeneratorWeights(
        w_hydrophobic_core=1.0,
        w_run_penalty=1.0,
        w_forbidden_pattern=2.0,
        w_charge_penalty=0.5,
    )

    gen_conf = GeneratorConfig(
        length=80,
        temperature=0.7,
        max_backtrack_steps=300,
        max_restarts=5,
        diversity_seed=42,
    )

    config = QIPCConfig(
        aa_constraints=aa_cons,
        codon_constraints=codon_cons,
        weights=weights,
        gen=gen_conf,
    )

    qipc = QIPCGenerator(config)

    protein = qipc.generate_protein_sequence()
    print("Protein sequence:", protein)

    if protein:
        dna = qipc.generate_dna_for_sequence(protein)
        print("DNA sequence:", dna)

