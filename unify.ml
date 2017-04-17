(* implementation of http://ropas.snu.ac.kr/lib/dock/Wac.ps *)
(* Type Inference for Objects with Instance Variables and Inheritance *)
(* Useful http://www.cs.cornell.edu/courses/cs312/2005sp/lectures/lec22.asp *)


type tree = 
      Var of string * int
    | Const of string
    | Absent
    | Present of tree
    | Arrow of tree * tree
    | Record of tree
    | Extension of (string * tree) list * tree
    | Empty




let tvar s = Var(s, 0)

let rec mapappend l = match l with
    | [] -> ""
    | (x :: y) -> (x ^ (mapappend y))

type node = 
      NullNode
    | VarNode of string * int * node ref
    | ConstNode of string
    | AbsentNode of node ref
    | PresentNode of node * node ref
    | ArrowNode of node * node * node ref
    | RecordNode of node * node ref
    | ExtensionNode of (string * node) list * node
    | EmptyNode of node ref


exception Link_of_Null of unit

let link n = match n with
    | NullNode -> raise (Link_of_Null ())
    | (ConstNode s) -> ref NullNode
    | (VarNode (s, i, l)) -> l
    | (AbsentNode (l)) -> l
    | (PresentNode (s, l)) -> l
    | (ArrowNode (n1,n2, x)) -> x
    | (RecordNode (_, l)) -> l
    | (ExtensionNode (_, _)) -> ref NullNode
    | (EmptyNode (l)) -> l

let gen_num : unit -> int = 
    let count = ref 0 in
    fun x -> (count := !count + 1; !count);;

let rec find n = if !(link n) = NullNode then n else find (!(link n));;

let equiv n n1 = 
    let r = (find n) in
    let r1 = (find n1) in
    (link r) := r1;;


let empty_table = [];;

let add_to_table table sym v = (sym, v) :: table;;

let rec lookup x env succ fail = match env with
    | ((y, v)::table) -> if x = y then (succ v) else (lookup x table succ fail)
    | [] -> fail();;

let rec translate v a  = match v with
    | (Var (s, i)) -> lookup v a (fun x -> (x, a))
                                    (fun () -> let n = VarNode (s, i, ref NullNode)
                                        in (n, add_to_table a v n))
    | (Const (s)) -> (ConstNode(s), a)
    | (Absent) -> (AbsentNode(ref NullNode), a)
    | (Present (t)) -> let (n, a) = translate t a 
                        in (PresentNode(n, ref NullNode), a)
    | (Arrow(t1, t2)) -> 
        let (n1, a1) = translate t1 a in
        let (n2, a2) = translate t2 a1
            in (ArrowNode(n1, n2, ref NullNode), a2)
    | (Record(t)) ->
        let (n, a) = translate t a
            in (RecordNode(n, ref NullNode), a)
    | (Extension(l,t)) ->
        let rec loop enve a = match enve with
            | [] -> ([], a)
            | ((s, t)::l) -> 
                let (n, a1) = translate t a in
                let (l, a2) = loop l a1 in
                    ((s, n)::l, a2) in 
        let (n1, a1) = translate t a in
        let (l, a2) = loop l a1
            in (ExtensionNode(l, n1), a2)
    | (Empty) -> (EmptyNode(ref NullNode), a);;


let pair_to_string f (s, t) = "[\"" ^ s ^ "\" " ^ f t ^ "]";;

let rec tree_to_string t = match t with
    | Var (s, i) ->  "(Var \"" ^ s ^ "\" " ^ string_of_int i ^ ")"
    | Const (s) -> "(Const " ^ s ^ ")"
    | Absent -> "Absent"
    | Present (t1) -> "(Present " ^ tree_to_string t1 ^ ")"
    | Arrow (t1, t2) -> "(Arrow " ^ tree_to_string t1 ^ " " ^ tree_to_string t2 ^ ")"
    | Record (t1) -> "(Record " ^ tree_to_string t1 ^ ")"
    | Extension (ts, t) -> "(Extension " ^ String.concat " " (List.map (pair_to_string(tree_to_string)) ts) ^ " " ^ tree_to_string t ^ ")"
    | Empty -> "Empty"


let rec member x coll = match coll with
    | [] -> false
    | (y :: l) -> if x == y then true else member x l;;

let rec untranslate = 
    let rec loop n seen = 
        if member n seen then Var("...", 0)
          else match n with
              | VarNode(s,i,r) -> Var(s, i)
              | ConstNode(s) -> Const(s)
              | AbsentNode(l) -> Absent
              | PresentNode(n1, l) -> Present(loop (find n1) (n::seen))
              | ArrowNode(n1, n2, r) -> Arrow(loop (find n1) (n::seen),
                                              loop (find n2) (n::seen))
              | RecordNode(t, l) -> Record(loop (find t) (n::seen))
              | ExtensionNode(l, n2) -> Extension(List.map (fun (s, n1) -> (s, loop (find n1) (n::seen)))
                                                    l,
                                                    loop (find n2) (n::seen))
              | EmptyNode(_) -> Empty
              | NullNode -> raise (Link_of_Null ())
    in (fun n -> (loop n []));;

let printn n = print_string ((tree_to_string (untranslate n)) ^ "\n");;

let printsl sl = 
    (print_string "[";
    (List.iter (fun s -> (print_string s; print_string ", ")) sl);
    print_string "]\n");;

exception Kinding_Failed of node


let rec explicits n = match n with
    | (ExtensionNode(pairs, ext)) -> List.append (List.map fst pairs) (explicits (find ext))
    | (VarNode (_,_,_)) -> []
    | n -> raise (Kinding_Failed n);;

let rec setdiff ls l2 = match ls with
    | [] -> []
    | (x::l1) -> if member x l2 then setdiff l1 l2 
                                else x::(setdiff l1 l2);;


let pad t newlabels newext = 
    let rec loop n = match n with
        | (ExtensionNode(_, t)) -> loop (find t)
        | (VarNode(_, _, _)) -> (equiv n (ExtensionNode
                                    (List.map (fun s -> (s, VarNode("f", gen_num(), ref NullNode)))
                                            newlabels,
                                        newext)))
        | (EmptyNode(_)) -> (equiv n (ExtensionNode
                                        (List.map (fun s -> (s, VarNode("f", gen_num(), ref NullNode)))
                                            newlabels,
                                        newext)))
        | n -> (printn n; raise (Kinding_Failed n))
    in loop (find t);;


let rec find_field s n = match n with
    | (ExtensionNode(pairs, ext)) -> lookup s pairs (fun f -> f) (fun () -> find_field s (find ext))
    | _ -> raise (Kinding_Failed NullNode);;

let subgoals r1 r2 = 
    let r1 = (find r1) and r2 = find r2 in
    let e1 = explicits r1
in List.map (fun s -> (find_field s r1, find_field s r2)) e1;;

exception Unify_Failed of node * node

let rec unify n1 n2 =
    let n1 = find n1 and n2 = find n2 in
        if n1 = n2 then ()
        else match (n1, n2) with
            | (VarNode(_, _, _), n2) -> equiv n1 n2
            | (ConstNode(s), ConstNode(t)) -> if s = t then () else raise (Unify_Failed (n1, n2))
            | (n1, VarNode(_, _, _)) -> equiv n2 n1
            | (AbsentNode(_), AbsentNode(_)) -> equiv n1 n2
            | (PresentNode(t1, _), PresentNode(t2, _)) -> (equiv n1 n2; unify t1 t2)
            | (ArrowNode(t1, t2, r), ArrowNode(u1, u2, s)) -> (equiv n1 n2; unify t1 u1; unify t2 u2)
            | (RecordNode(t1, _), RecordNode(t2,_)) -> (equiv n1 n2; unify_row t1 t2)
            | (ExtensionNode(_, _), _) -> (printn n1; raise (Kinding_Failed n1))
            | (_, ExtensionNode(_, _)) -> (printn n2; raise (Kinding_Failed n2))
            | n -> raise (Unify_Failed (n1, n2))
and unify_row t1 t2 =
    let exp1 = explicits t1 and exp2 = explicits t2 in
    let a = setdiff exp1 exp2 and b = setdiff exp2 exp1 in
    let r = VarNode("r", gen_num(), ref NullNode) in
        (pad t1 b r; pad t2 a r;
            List.iter (fun (n1, n2) -> unify n1 n2) (subgoals t1 t2);
            ());;

let unify1 n1 n2 = try unify n1 n2 with
    | (Unify_Failed (n1, n2)) -> (print_string "Unify_Failed: \n";
                                     printn n1; printn n2; ())
    | (Kinding_Failed (n)) -> (print_string "Unify_Failed: \n";
                                     printn n;  ())






let testit () =
    let (n1, a) = translate (Arrow (tvar "x", Arrow (tvar "y", tvar "y"))) [] in
    let (n2, a) = translate (Arrow (tvar "x", tvar "x")) a
    in (unify1 n1 n2;
        print_string (tree_to_string (untranslate n1));
        print_string "\n"; true);;

let test2 () =
    let (n1, a) = translate (Arrow (tvar "x", tvar "x")) [] in
    let (n2, a) = translate (tvar "x") a
    in (unify1 n1 n2;
        print_string (tree_to_string (untranslate n1));
        print_string "\n"; true);;


let make2 () = 
    let (n1, a) = translate (Record (Extension([("a", Present(Const "bool"))], tvar "r1"))) [] in
    let (n2, a) = translate (Record (Extension([("a", Present(Const "int"))], tvar "r2"))) []
in (n1, n2);;



let (-->) a b = (Arrow (tvar a, tvar b))

let uni t1 t2 = 
    let (n1, a) = translate t1 [] in
    let (n2, a) = translate t2 a in
        (unify1 n1 n2);;


let apply f x = match f with
    | (Arrow (a, b)) -> (uni a x; b)
    | _ -> raise (Kinding_Failed NullNode) ;;

let ident = (Arrow (Const "bool", Const "int"))
let x = (Const "bool");;

(* uni ident x;; *)

let q = apply ident x;;
(* print_string (tree_to_string q) *)


let test3 () = let (n1, n2) = make2 () in
    (unify1 n1 n2;
        printn n1);;


(* test3 () *)

type expr =
    True
  | False
  | Num of int
  | If of expr * expr * expr

let rec tinfer e = match e with
    | True -> (Const "bool")
    | False -> (Const "bool")
    | Num(i) -> (Const "int")
    | If(e1, e2, e3) ->
        let t1 = tinfer(e1) in
        let t2 = tinfer(e2) in
        let t3 = tinfer(e3)
      in
        uni t1 (Const "bool");
        uni t2 t3;
        t2;;


print_string (tree_to_string (tinfer (If (True, (Num 2), (Num 1)))))







