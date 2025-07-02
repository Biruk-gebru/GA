%% Generated from /home/karanos/gatask/GA/ga_operators.metta at 2025-07-02T21:36:09+03:00
:- style_check(-discontiguous).
:- style_check(-singleton).
:- include(library(metta_lang/metta_transpiled_header)).
:- set_prolog_flag(pfc_term_expansion,true).

%  ;; Idiomatic MeTTa: GA Operators
%  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
%  ;; Helper: selectByIndex
%  ;; Helper: size of tuple
%  ;; Helper: isMember
%  ;; Helper: cons unique (for tournament sampling)
%  ;; Helper: sample $k unique indices from 0 to $n-1
%  ;; Helper: argmax over indices
%  ;; $indices: tuple of indices, $fitnesses: tuple of fitnesses
%  ;; Tournament Selection
%  ;; Uniform Crossover
%  ;; Polynomial Mutation
%  ;; Helper: reverse a tuple
%  ;; Non-deterministic Roulette-Wheel Selection
%  ;; Helper: sum of a tuple
%  ;; Example Calls
%  ;; Suppose:
%  ; ;; Tournament selection (k=2):
%  ; !(tournament-selection pop fits 2)
%  ; ;; Uniform crossover:
%  ; !(uniform-crossover (0.1 0.2 0.3) (0.4 0.5 0.6))
%  ; ;; Polynomial mutation (eta=20, mutation-rate=0.1):
%  ; !(polynomial-mutation (0.1 0.2 0.3) 20 0.1)
%  ; ;; Roulette-wheel selection:
%  ; !(roulette-wheel-selection pop fits)
%  ;; Main GA Loop
%  ;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
%  ;; Main GA Loop (Revised)
%  ;; Wrapper for roulette selection to match tournament interface
%  ;; Helper: create initial population
%  ;; Helper: create random individual
%  ;; Helper: create range [0, n)
%  ;; Main GA function
%  ;; Revised ga-helper without 'do'
%  ;; Create offspring population
%  ;; Helper: map function over list
%  ;; Example Runs
%  ;; Print messages and run GA
%  ;; Execute both configurations


<span class="pl-atom">top_call_5</span>:- <span class="pl-atom">do_metta_runtime</span><span class="pl-functor">( <span class="pl-var">ExecRes</span>, </span>
                                              ((&#13;&#10; 
                                               (<span class="pl-atom">true</span>),
                                               (<span class="pl-atom">me</span>( <span class="pl-atom">'run-ga'</span>, 
                                                  <span class="pl-string">"Running GA with Tournament Selection, Uniform Crossover, Polynomial Mutation ---"</span>, <span class="pl-atom">'tournament-selection'</span>, <span class="pl-var">ExecRes</span>))  ))).




top_call :-
    time(top_call_5).


==>arg_type_n('init-population-helper',3,3,non_eval('Expression')).
==>arg_type_n('ga-helper',11,1,non_eval([->,'Expression','Expression','Number','Atom'])).
==>arg_type_n('ga-helper',11,2,non_eval([->,'Expression','Expression',['Expression','Expression']])).
==>arg_type_n('ga-helper',11,3,non_eval([->,'Expression','Number','Number','Expression'])).
==>arg_type_n('ga-helper',11,4,non_eval('Expression')).
==>arg_type_n(==,2,1,var).
==>arg_type_n(==,2,2,var).
==>arg_type_n(map,2,1,non_eval([->,'Expression','Number'])).
==>arg_type_n(map,2,2,non_eval('Expression')).
==>arg_type_n(sum,1,1,non_eval('Expression')).
