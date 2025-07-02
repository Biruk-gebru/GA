;; Idiomatic MeTTa: GA Operators
;; NOTE: The 'size' helper is now a wrapper for the stdlib 'size-atom'.
;; NOTE: The 'map' helper is removed; use stdlib 'map-atom'.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Helper: selectByIndex
(: selectByIndex (-> Expression Number Atom))
(= (selectByIndex $expr $index)
    (if (== $index 0)
        (car-atom $expr)
        (selectByIndex (cdr-atom $expr) (- $index 1))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Helper: size of tuple (delegates to stdlib size-atom)
(: size (-> Expression Number))
(= (size $xs)
    (size-atom $xs))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Helper: isMember
(: isMember (-> $a $aa Bool))
(= (isMember $x $list)
    (not (== (collapse (intersection (superpose ($x)) (superpose $list))) ()))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Helper: cons unique (for tournament sampling)
(: cons-unique (-> Atom Expression Expression))
(= (cons-unique $x $xs)
    (if (isMember $x $xs) $xs (cons-atom $x $xs)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Helper: sample $k unique indices from 0 to $n-1
(: tournament-sample (-> Number Number Expression Expression))
(= (tournament-sample $n 0 $acc) $acc)
(= (tournament-sample $n $k $acc)
    (let* (
        ($range (- $n 1))
        ($idx (random-int &rng 0 $range))
        ($next (cons-unique $idx $acc))
        ($nextk (if (isMember $idx $acc) $k (- $k 1)))
    )
        (tournament-sample $n $nextk $next)
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Helper: argmax-index
;; Given a tuple of indices and a tuple of fitnesses, return the index with the highest fitness.
(: argmax-index (-> Expression Expression Number))
(= (argmax-index $indices $fitnesses)
    (argmax-index-helper (cdr-atom $indices) $fitnesses (car-atom $indices) (selectByIndex $fitnesses (car-atom $indices)))
)
(: argmax-index-helper (-> Expression Expression Number Number Number))
(= (argmax-index-helper $indices $fitnesses $best-idx $best-val)
    (if (== $indices ())
        $best-idx
        (let* (
            ($i (car-atom $indices))
            ($val (selectByIndex $fitnesses $i))
            ($new-best-idx (if (> $val $best-val) $i $best-idx))
            ($new-best-val (if (> $val $best-val) $val $best-val))
        )
            (argmax-index-helper (cdr-atom $indices) $fitnesses $new-best-idx $new-best-val)
        )
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Tournament selection: returns the individual from $population with the highest fitness among $k random samples.
(: tournament-selection (-> Expression Expression Number Atom))
(= (tournament-selection $population $fitnesses $k)
    (let* (
        ($n (size $population))
        ($indices (tournament-sample $n $k ()))
        ($best-idx (argmax-index $indices $fitnesses))
    )
        (selectByIndex $population $best-idx)
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Uniform Crossover
(: uniform-crossover (-> Expression Expression (Expression Expression)))
(= (uniform-crossover $p1 $p2)
    (uniform-crossover-helper $p1 $p2 () ())
)
(: uniform-crossover-helper (-> Expression Expression Expression Expression (Expression Expression)))
(= (uniform-crossover-helper $p1 $p2 $acc1 $acc2)
    (if (== $p1 ())
        ((reverse $acc1) (reverse $acc2))
        (let* (
            ($g1 (car-atom $p1))
            ($g2 (car-atom $p2))
            ($rest1 (cdr-atom $p1))
            ($rest2 (cdr-atom $p2))
            ($r (random-float &rng 0 1))
            ($new1 (if (< $r 0.5) (cons-atom $g1 $acc1) (cons-atom $g2 $acc1)))
            ($new2 (if (< $r 0.5) (cons-atom $g2 $acc2) (cons-atom $g1 $acc2)))
        )
            (uniform-crossover-helper $rest1 $rest2 $new1 $new2)
        )
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Polynomial Mutation
(: polynomial-mutation (-> Expression Number Number Expression))
(= (polynomial-mutation $ind $eta $mutation-rate)
    (polynomial-mutation-helper $ind $eta $mutation-rate ())
)
(: polynomial-mutation-helper (-> Expression Number Number Expression Expression))
(= (polynomial-mutation-helper $ind $eta $mutation-rate $acc)
    (if (== $ind ())
        (reverse $acc)
        (let* (
            ($gene (car-atom $ind))
            ($r1 (random-float &rng 0 1))
            ($mutated
                (if (< $r1 $mutation-rate)
                    (let* (
                        ($r2 (random-float &rng 0 1))
                        ($delta (if (< $r2 0.5)
                            (- (pow-math (* 2 $r2) (/ 1 (+ $eta 1))) 1)
                            (- 1 (pow-math (* 2 (- 1 $r2)) (/ 1 (+ $eta 1))))
                        ))
                        ($mut (+ $gene $delta))
                        ($clipped (if (< $mut 0) 0 (if (> $mut 1) 1 $mut)))
                    ) $clipped)
                    $gene
                )
            )
        )
            (polynomial-mutation-helper (cdr-atom $ind) $eta $mutation-rate (cons-atom $mutated $acc))
        )
    )
)

;; Helper: reverse a tuple
(: reverse (-> Expression Expression))
(= (reverse $xs)
    (reverse-helper $xs ()))
(: reverse-helper (-> Expression Expression Expression))
(= (reverse-helper $xs $acc)
    (if (== $xs ()) $acc (reverse-helper (cdr-atom $xs) (cons-atom (car-atom $xs) $acc))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Non-deterministic Roulette-Wheel Selection (superpose version)
;; This version uses superpose to enumerate all possible selections.
;; It is useful for symbolic or exhaustive search, but not needed for standard probabilistic selection.
;;
;; (: roulette-superpose (-> Expression Expression Atom))
;; (= (roulette-superpose $population $fitnesses)
;;     (superpose
;;         (roulette-helper $population $fitnesses (random-float &rng 0 (sum $fitnesses)) 0 0)
;;     )
;; )
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Non-deterministic Roulette-Wheel Selection
(: roulette-wheel-selection (-> Expression Expression Atom))
(= (roulette-wheel-selection $population $fitnesses)
    (let* (
        ($total (sum $fitnesses))
        ($r (random-float &rng 0 $total))
    )
        (roulette-helper $population $fitnesses $r 0 0)
    )
)
(: roulette-helper (-> Expression Expression Number Number Number Atom))
(= (roulette-helper $pop $fits $r $i $acc)
    (if (>= (+ $acc (selectByIndex $fits $i)) $r)
        (selectByIndex $pop $i)
        (roulette-helper $pop $fits $r (+ $i 1) (+ $acc (selectByIndex $fits $i)))
    )
)

;; Helper: sum of a tuple
(: sum (-> Expression Number))
(= (sum $xs)
    (if (== $xs ()) 0 (+ (car-atom $xs) (sum (cdr-atom $xs)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Fitness function: sum of genes (can be replaced with a custom function)
(: fitness (-> Expression Number))
(= (fitness $ind) (sum $ind))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Example Calls
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Suppose:
(bind! pop ((0.1 0.2 0.3) (0.4 0.5 0.6) (0.7 0.8 0.9)))
(bind! fits (0.5 0.7 0.2))

; ;; Tournament selection (k=2):

; ;; Uniform crossover:
; !(uniform-crossover (0.1 0.2 0.3) (0.4 0.5 0.6))

; ;; Polynomial mutation (eta=20, mutation-rate=0.1):
; !(polynomial-mutation (0.1 0.2 0.3) 20 0.1)

; ;; Roulette-wheel selection:
; !(roulette-wheel-selection pop fits)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Main GA Loop
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Main GA Loop (Revised)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Wrapper for roulette selection to match tournament interface
(: roulette-wrapper (-> Expression Expression Number Atom))
(= (roulette-wrapper $pop $fits $ignored) 
   (roulette-wheel-selection $pop $fits))

;; Helper: create initial population
(: init-population (-> Number Number Expression))
(= (init-population $pop-size $num-genes)
   (init-population-helper $pop-size $num-genes ()))
(: init-population-helper (-> Number Number Expression Expression))
(= (init-population-helper 0 $num-genes $acc) $acc)
(= (init-population-helper $n $num-genes $acc)
   (init-population-helper (- $n 1) $num-genes 
        (cons-atom (random-individual $num-genes) $acc)))

;; Helper: create random individual
(: random-individual (-> Number Expression))
(= (random-individual 0) ())
(= (random-individual $n) 
   (cons-atom (random-float &rng 0 1) (random-individual (- $n 1))))

;; Helper: create range [0, n)
(: range (-> Number Expression))
(= (range 0) ())
(= (range $n) 
   (if (> $n 0) (cons-atom (- $n 1) (range (- $n 1))) ()))

;; Main GA function
(: genetic-algorithm (-> (-> Expression Expression Number Atom) 
                         (-> Expression Expression (Expression Expression)) 
                         (-> Expression Number Number Expression)
                         Number Number Number Number Number ()))
(= (genetic-algorithm $sel-fn $cross-fn $mut-fn $pop-size $num-genes $num-gen $k $eta $mut-rate)
   (let* (
        ($pop (init-population $pop-size $num-genes))
        ($result (ga-helper $sel-fn $cross-fn $mut-fn $pop $pop-size $num-genes $num-gen $k $eta $mut-rate 0))
    )
    ()
   )
)

;; Revised ga-helper without 'do'
(: ga-helper (-> (-> Expression Expression Number Atom) 
                 (-> Expression Expression (Expression Expression)) 
                 (-> Expression Number Number Expression)
                 Expression Number Number Number Number Number Number Number
                 ()))
(= (ga-helper $sel-fn $cross-fn $mut-fn $pop $pop-size $num-genes $num-gen $k $eta $mut-rate $gen)
   (if (== $gen $num-gen)
      (let* (
            ($fits (map-atom fitness $pop))
            ($best-idx (argmax (range $pop-size) $fits))
            ($_ (println "Final best individual:" (selectByIndex $pop $best-idx) 
                         "with fitness:" (selectByIndex $fits $best-idx)))
        )
        ()
      )
      (let* (
            ($fits (map-atom fitness $pop))
            ($best-idx (argmax (range $pop-size) $fits))
            ($_ (println "Generation:" $gen 
                         "Best fitness:" (selectByIndex $fits $best-idx)))
            ($next-pop (make-offspring $sel-fn $cross-fn $mut-fn $pop $fits $pop-size $k $eta $mut-rate ()))
        )
        (ga-helper $sel-fn $cross-fn $mut-fn $next-pop $pop-size $num-genes $num-gen $k $eta $mut-rate (+ $gen 1))
      )
   )
)

;; Create offspring population
(: make-offspring (-> (-> Expression Expression Number Atom) 
                      (-> Expression Expression (Expression Expression)) 
                      (-> Expression Number Number Expression)
                      Expression Expression Number Number Number Number Expression 
                      Expression))
(= (make-offspring $sel-fn $cross-fn $mut-fn $pop $fits $remaining $k $eta $mut-rate $acc)
   (if (== $remaining 0)
      $acc
      (let* (
         ;; Select parents
         ($p1 ($sel-fn $pop $fits $k))
         ($p2 ($sel-fn $pop $fits $k))
         
         ;; Crossover
         ($children ($cross-fn $p1 $p2))
         ($c1 (car-atom $children))
         ($c2 (car-atom (cdr-atom $children)))
         
         ;; Mutate
         ($c1m ($mut-fn $c1 $eta $mut-rate))
         ($c2m ($mut-fn $c2 $eta $mut-rate))
      )
         (make-offspring $sel-fn $cross-fn $mut-fn $pop $fits (- $remaining 2) $k $eta $mut-rate 
                         (cons-atom $c1m (cons-atom $c2m $acc)))
      )
   )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Example Runs
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Print messages and run GA
(= (run-ga $sel-fn)
   (let* (
        ; ($_ (println! ($message)))
        ($result (genetic-algorithm $sel-fn uniform-crossover polynomial-mutation 100 3 20 2 20 0.1))
    )
    ()
   )
)


; !(tournament-selection ((0.1 0.2 0.3) (0.4 0.5 0.6) (0.7 0.8 0.9)) (0.5 0.7 0.2) 2)
!====1====
!(uniform-crossover (0.1 0.2 0.3) (0.4 0.5 0.6))
!====2==
!(polynomial-mutation (0.1 0.2 0.3) 20 0.1)

! =======
;; Execute both configurations
!("Running GA with Tournament Selection, Uniform Crossover, Polynomial Mutation ---")
!(run-ga tournament-selection)


!("Running GA with Roulette-Wheel Selection, Uniform Crossover, Polynomial Mutation ---")
!(run-ga roulette-wrapper)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test call for tournament-selection with actual dummy values
;; population = ((0.1 0.2 0.3) (0.4 0.5 0.6) (0.7 0.8 0.9))
;; fitnesses = (0.5 0.7 0.2)
;; k = 2
; !(tournament-selection ((0.1 0.2 0.3) (0.4 0.5 0.6) (0.7 0.8 0.9)) (0.5 0.7 0.2) 2)