axiom Absolute : Type
axiom magnitude : Absolute → ℝ
axiom direction : Absolute → ℤ
def add (a b : Absolute) : Absolute := a
def mul (a b : Absolute) : Absolute := a
axiom comm_add : ∀ a b : Absolute, add a b = add b a
axiom compensation : ∀ a b : Absolute, magnitude a = magnitude b → direction a = - direction b → magnitude (add a b) = 0
lemma add_comm (a b : Absolute) : add a b = add b a := comm_add a b
