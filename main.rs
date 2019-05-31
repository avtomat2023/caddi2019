pub trait Readable {
    type Output;
    fn words_count() -> usize;
    fn read_words(words: &[&str]) -> Result<Self::Output, String>;
}
#[macro_export]
macro_rules! readable {
    ( $ t : ty , $ words_count : expr , |$ words : ident | $ read_words : expr ) => {
        impl Readable for $t {
            type Output = $t;
            fn words_count() -> usize {
                $words_count
            }
            fn read_words($words: &[&str]) -> Result<$t, String> {
                Ok($read_words)
            }
        }
    };
}
readable!((), 1, |_ss| ());
readable!(String, 1, |ss| ss[0].to_string());
impl Readable for char {
    type Output = char;
    fn words_count() -> usize {
        1
    }
    fn read_words(words: &[&str]) -> Result<char, String> {
        let chars: Vec<char> = words[0].chars().collect();
        if chars.len() == 1 {
            Ok(chars[0])
        } else {
            Err(format!("cannot parse \"{}\" as a char", words[0]))
        }
    }
}
pub struct Chars();
impl Readable for Chars {
    type Output = Vec<char>;
    fn words_count() -> usize {
        1
    }
    fn read_words(words: &[&str]) -> Result<Vec<char>, String> {
        Ok(words[0].chars().collect())
    }
}
macro_rules ! impl_readable_for_ints { ( $ ( $ t : ty ) * ) => { $ ( impl Readable for $ t { type Output = Self ; fn words_count ( ) -> usize { 1 } fn read_words ( words : & [ & str ] ) -> Result <$ t , String > { use std :: str :: FromStr ; <$ t >:: from_str ( words [ 0 ] ) . map_err ( | _ | { format ! ( "cannot parse \"{}\" as {}" , words [ 0 ] , stringify ! ( $ t ) ) } ) } } ) * } ; }
impl_readable_for_ints ! ( i8 u8 i16 u16 i32 u32 i64 u64 isize usize f32 f64 ) ;
macro_rules ! define_one_origin_int_types { ( $ new_t : ident $ int_t : ty ) => { # [ doc = " Converts 1-origin integer into 0-origin when read from stdin." ] # [ doc = "" ] # [ doc = " # Example" ] # [ doc = "" ] # [ doc = " ```no_run" ] # [ doc = " # #[macro_use] extern crate atcoder_snippets;" ] # [ doc = " # use atcoder_snippets::read::*;" ] # [ doc = " // Stdin: \"1\"" ] # [ doc = " read!(a = usize_);" ] # [ doc = " assert_eq!(a, 0);" ] # [ doc = " ```" ] # [ allow ( non_camel_case_types ) ] pub struct $ new_t ( ) ; impl Readable for $ new_t { type Output = $ int_t ; fn words_count ( ) -> usize { 1 } fn read_words ( words : & [ & str ] ) -> Result < Self :: Output , String > { <$ int_t >:: read_words ( words ) . map ( | n | n - 1 ) } } } ; ( $ new_t : ident $ int_t : ty ; $ ( $ inner_new_t : ident $ inner_int_t : ty ) ;* ) => { define_one_origin_int_types ! ( $ new_t $ int_t ) ; define_one_origin_int_types ! ( $ ( $ inner_new_t $ inner_int_t ) ;* ) ; } ; }
define_one_origin_int_types ! ( u8_ u8 ; u16_ u16 ; u32_ u32 ; u64_ u64 ; usize_ usize ) ;
macro_rules ! impl_readable_for_tuples { ( $ t : ident $ var : ident ) => ( ) ; ( $ t : ident $ var : ident ; $ ( $ inner_t : ident $ inner_var : ident ) ;* ) => { impl_readable_for_tuples ! ( $ ( $ inner_t $ inner_var ) ;* ) ; impl <$ t : Readable , $ ( $ inner_t : Readable ) ,*> Readable for ( $ t , $ ( $ inner_t ) ,* ) { type Output = ( <$ t >:: Output , $ ( <$ inner_t >:: Output ) ,* ) ; fn words_count ( ) -> usize { let mut n = <$ t >:: words_count ( ) ; $ ( n += <$ inner_t >:: words_count ( ) ; ) * n } # [ allow ( unused_assignments ) ] fn read_words ( words : & [ & str ] ) -> Result < Self :: Output , String > { let mut start = 0 ; let $ var = <$ t >:: read_words ( & words [ start .. start +<$ t >:: words_count ( ) ] ) ?; start += <$ t >:: words_count ( ) ; $ ( let $ inner_var = <$ inner_t >:: read_words ( & words [ start .. start +<$ inner_t >:: words_count ( ) ] ) ?; start += <$ inner_t >:: words_count ( ) ; ) * Ok ( ( $ var , $ ( $ inner_var ) ,* ) ) } } } ; }
impl_readable_for_tuples ! ( T4 x4 ; T3 x3 ; T2 x2 ; T1 x1 ) ;
pub trait ReadableFromLine {
    type Output;
    fn read_line(line: &str) -> Result<Self::Output, String>;
}
fn split_into_words(line: &str) -> Vec<&str> {
    #[allow(deprecated)]
    line.trim_right_matches('\n').split_whitespace().collect()
}
impl<T: Readable> ReadableFromLine for T {
    type Output = T::Output;
    fn read_line(line: &str) -> Result<T::Output, String> {
        let words = split_into_words(line);
        if words.len() != T::words_count() {
            return Err(format!(
                "line \"{}\" has {} words, expected {}",
                line,
                words.len(),
                T::words_count()
            ));
        }
        T::read_words(&words)
    }
}
impl<T: Readable> ReadableFromLine for Vec<T> {
    type Output = Vec<T::Output>;
    fn read_line(line: &str) -> Result<Self::Output, String> {
        let n = T::words_count();
        let words = split_into_words(line);
        if words.len() % n != 0 {
            return Err(format!(
                "line \"{}\" has {} words, expected multiple of {}",
                line,
                words.len(),
                n
            ));
        }
        let mut result = Vec::new();
        for chunk in words.chunks(n) {
            match T::read_words(chunk) {
                Ok(v) => result.push(v),
                Err(msg) => {
                    let flagment_msg = if n == 1 {
                        format!("word {}", result.len())
                    } else {
                        let l = result.len();
                        format!("words {}-{}", n * l + 1, (n + 1) * l)
                    };
                    return Err(format!("{} of line \"{}\": {}", flagment_msg, line, msg));
                }
            }
        }
        Ok(result)
    }
}
impl<T: Readable, U: Readable> ReadableFromLine for (T, Vec<U>) {
    type Output = (T::Output, <Vec<U> as ReadableFromLine>::Output);
    fn read_line(line: &str) -> Result<Self::Output, String> {
        let n = T::words_count();
        #[allow(deprecated)]
        let trimmed = line.trim_right_matches('\n');
        let words_and_rest: Vec<&str> = trimmed.splitn(n + 1, ' ').collect();
        if words_and_rest.len() < n {
            return Err(format!(
                "line \"{}\" has {} words, expected at least {}",
                line,
                words_and_rest.len(),
                n
            ));
        }
        let words = &words_and_rest[..n];
        let empty_str = "";
        let rest = words_and_rest.get(n).unwrap_or(&empty_str);
        Ok((T::read_words(words)?, Vec::<U>::read_line(rest)?))
    }
}
macro_rules ! impl_readable_from_line_for_tuples_with_vec { ( $ t : ident $ var : ident ) => ( ) ; ( $ t : ident $ var : ident ; $ ( $ inner_t : ident $ inner_var : ident ) ;+ ) => { impl_readable_from_line_for_tuples_with_vec ! ( $ ( $ inner_t $ inner_var ) ;+ ) ; impl <$ t : Readable , $ ( $ inner_t : Readable ) ,+ , U : Readable > ReadableFromLine for ( $ t , $ ( $ inner_t ) ,+ , Vec < U > ) { type Output = ( $ t :: Output , $ ( $ inner_t :: Output ) ,+ , Vec < U :: Output > ) ; fn read_line ( line : & str ) -> Result < Self :: Output , String > { let mut n = $ t :: words_count ( ) ; $ ( n += $ inner_t :: words_count ( ) ; ) + # [ allow ( deprecated ) ] let trimmed = line . trim_right_matches ( '\n' ) ; let words_and_rest : Vec <& str > = trimmed . splitn ( n + 1 , ' ' ) . collect ( ) ; if words_and_rest . len ( ) < n { return Err ( format ! ( "line \"{}\" has {} words, expected at least {}" , line , words_and_rest . len ( ) , n ) ) ; } let words = & words_and_rest [ .. n ] ; let empty_str = "" ; let rest = words_and_rest . get ( n ) . unwrap_or ( & empty_str ) ; let ( $ var , $ ( $ inner_var ) ,* ) = < ( $ t , $ ( $ inner_t ) ,+ ) >:: read_words ( words ) ?; Ok ( ( $ var , $ ( $ inner_var ) ,* , Vec ::< U >:: read_line ( rest ) ? ) ) } } } ; }
impl_readable_from_line_for_tuples_with_vec ! ( T4 t4 ; T3 t3 ; T2 t2 ; T1 t1 ) ;
pub fn read<T: ReadableFromLine>() -> T::Output {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    T::read_line(&line).unwrap()
}
#[macro_export]
macro_rules ! read { ( ) => { let mut line = String :: new ( ) ; std :: io :: stdin ( ) . read_line ( & mut line ) . unwrap ( ) ; } ; ( $ pat : pat = $ t : ty ) => { let $ pat = read ::<$ t > ( ) ; } ; ( $ ( $ pat : pat = $ t : ty ) ,+ ) => { read ! ( ( $ ( $ pat ) ,* ) = ( $ ( $ t ) ,* ) ) ; } ; }
#[macro_export]
macro_rules ! readls { ( $ ( $ pat : pat = $ t : ty ) ,+ ) => { $ ( read ! ( $ pat = $ t ) ; ) * } ; }
pub fn readx<T: ReadableFromLine>() -> Vec<T::Output> {
    use std::io::{self, BufRead};
    let stdin = io::stdin();
    let result = stdin
        .lock()
        .lines()
        .map(|line_result| {
            let line = line_result.expect("read from stdin failed");
            T::read_line(&line).unwrap()
        }).collect();
    result
}
#[macro_export]
macro_rules ! readx_loop { ( |$ pat : pat = $ t : ty | $ body : expr ) => { use std :: io :: BufRead ; let stdin = std :: io :: stdin ( ) ; for line in stdin . lock ( ) . lines ( ) { let line = line . expect ( "read from stdin failed" ) ; let $ pat = <$ t >:: read_line ( & line ) . unwrap ( ) ; $ body } } ; ( |$ ( $ pat : pat = $ t : ty ) ,*| $ body : expr ) => { readx_loop ! ( | ( $ ( $ pat ) ,* ) = ( $ ( $ t ) ,* ) | $ body ) ; } ; }
pub fn readn<T: ReadableFromLine>(n: usize) -> Vec<T::Output> {
    use std::io::{self, BufRead};
    let stdin = io::stdin();
    let result: Vec<T::Output> = stdin
        .lock()
        .lines()
        .take(n)
        .map(|line_result| {
            let line = line_result.expect("read from stdin failed");
            T::read_line(&line).unwrap()
        }).collect();
    if result.len() < n {
        panic!(
            "expected reading {} lines, but only {} lines are read",
            n,
            result.len()
        );
    }
    result
}
#[macro_export]
macro_rules ! readn_loop { ( $ n : expr , |$ pat : pat = $ t : ty | $ body : expr ) => { use std :: io :: BufRead ; let stdin = std :: io :: stdin ( ) ; { let mut lock = stdin . lock ( ) ; for _ in 0 ..$ n { let mut line = String :: new ( ) ; lock . read_line ( & mut line ) . expect ( "read from stdin failed" ) ; let $ pat = <$ t >:: read_line ( & line ) . unwrap ( ) ; $ body } } } ; ( $ n : expr , |$ ( $ pat : pat = $ t : ty ) ,*| $ body : expr ) => { readn_loop ! ( $ n , | ( $ ( $ pat ) ,* ) = ( $ ( $ t ) ,* ) | $ body ) ; } ; }
pub trait Words {
    fn read<T: Readable>(&self) -> T::Output;
}
impl<'a> Words for [&'a str] {
    fn read<T: Readable>(&self) -> T::Output {
        T::read_words(self).unwrap()
    }
}
impl<'a> Words for &'a str {
    fn read<T: Readable>(&self) -> T::Output {
        T::read_words(&[self]).unwrap()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}
impl<T> Vec3<T> {
    pub fn new(x: T, y: T, z: T) -> Vec3<T> {
        Vec3 { x: x, y: y, z: z }
    }
}
impl<T: std::fmt::Display> std::fmt::Display for Vec3<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {} {}", self.x, self.y, self.z)
    }
}
impl<S, T: std::ops::Add<S>> std::ops::Add<Vec3<S>> for Vec3<T> {
    type Output = Vec3<T::Output>;
    fn add(self, rhs: Vec3<S>) -> Self::Output {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}
impl<S, T: std::ops::AddAssign<S>> std::ops::AddAssign<Vec3<S>> for Vec3<T> {
    fn add_assign(&mut self, rhs: Vec3<S>) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}
impl<S, T: std::ops::Sub<S>> std::ops::Sub<Vec3<S>> for Vec3<T> {
    type Output = Vec3<T::Output>;
    fn sub(self, rhs: Vec3<S>) -> Self::Output {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}
impl<S, T: std::ops::SubAssign<S>> std::ops::SubAssign<Vec3<S>> for Vec3<T> {
    fn sub_assign(&mut self, rhs: Vec3<S>) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}
impl<T: std::ops::Neg> std::ops::Neg for Vec3<T> {
    type Output = Vec3<T::Output>;
    fn neg(self) -> Self::Output {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}
macro_rules ! impl_mul_vec3 { ( $ ( $ t : ty ) * ) => { $ ( impl std :: ops :: Mul < Vec3 <$ t >> for $ t { type Output = Vec3 <<$ t as std :: ops :: Mul >:: Output >; fn mul ( self , rhs : Vec3 <$ t > ) -> Self :: Output { Vec3 :: new ( self * rhs . x , self * rhs . y , self * rhs . z ) } } ) * } }
impl_mul_vec3 ! ( i8 u8 i16 u16 i32 u32 i64 u64 f32 f64 ) ;
impl<S: Copy, T: std::ops::Mul<S>> std::ops::Mul<S> for Vec3<T> {
    type Output = Vec3<T::Output>;
    fn mul(self, rhs: S) -> Self::Output {
        Vec3::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}
impl<S: Copy, T: std::ops::MulAssign<S>> std::ops::MulAssign<S> for Vec3<T> {
    fn mul_assign(&mut self, rhs: S) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}
impl<S: Copy, T: std::ops::Div<S>> std::ops::Div<S> for Vec3<T> {
    type Output = Vec3<T::Output>;
    fn div(self, rhs: S) -> Self::Output {
        Vec3::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}
impl<S: Copy, T: std::ops::DivAssign<S>> std::ops::DivAssign<S> for Vec3<T> {
    fn div_assign(&mut self, rhs: S) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}
impl<T: Readable> Readable for Vec3<T> {
    type Output = Vec3<T::Output>;
    fn words_count() -> usize {
        T::words_count() * 3
    }
    fn read_words(words: &[&str]) -> Result<Vec3<T::Output>, String> {
        let n = T::words_count();
        Ok(Vec3::new(
            T::read_words(&words[..n])?,
            T::read_words(&words[n..2 * n])?,
            T::read_words(&words[2 * n..])?,
        ))
    }
}

const LENGTH: u32 = 1_000;
const SPHERE_COUNT: usize = 1_000;
const BONUS_COUNT: usize = 100_000;

mod data {
    use super::{Vec3, SPHERE_COUNT};
    use std::fmt::{self, Display, Formatter};
    use std::iter::FromIterator;
    use std::f64;

    #[derive(Debug)]
    pub struct Sphere {
        index: u16,
        radius: u32,
        point: u64,
        point_per_volume: f64,
    }

    impl Sphere {
        pub fn new(index: u16, radius: u32, point: u64) -> Sphere {
            let volume = (radius as f64).powi(3) * 4.0*f64::consts::PI / 3.0;
            Sphere {
                index: index,
                radius: radius,
                point: point,
                point_per_volume: (point as f64) / volume
            }
        }

        pub fn index(&self) -> u16 { self.index }
        pub fn radius(&self) -> u32 { self.radius }
        pub fn point(&self) -> u64 { self.point }
        pub fn point_per_volume(&self) -> f64 { self.point_per_volume }
    }

    #[derive(Clone, Debug)]
    struct BonusOpponent {
        index: u16,
        distance: u32,
        point: u64
    }

    #[derive(Debug)]
    pub struct BonusList {
        bonuses: Vec<Vec<BonusOpponent>>
    }

    impl FromIterator<(u16, u16, u32, u64)> for BonusList {
        fn from_iter<I: IntoIterator<Item = (u16, u16, u32, u64)>>(iter: I) -> BonusList {
            let mut bonuses = vec![Vec::new(); SPHERE_COUNT];
            for (a, b, dist, point) in iter {
                bonuses[a as usize].push(BonusOpponent {
                    index: b, distance: dist, point: point
                });
                bonuses[b as usize].push(BonusOpponent {
                    index: a, distance: dist, point: point
                });
            }
            BonusList { bonuses: bonuses }
        }
    }

    pub struct Knapsack {
        spheres: Vec<Option<Vec3<u16>>>
    }

    impl Knapsack {
        pub fn new(sphere_count: usize) -> Knapsack {
            Knapsack { spheres: vec![None; sphere_count] }
        }

        pub fn push_if_possible(
            &mut self,
            index: u16,
            pos: Vec3<u16>,
            spheres: &[Sphere],
            bonus_list: &BonusList
        ) -> bool {
            unimplemented!()
        }

        fn check_collision(
            &self,
            index: u16,
            pos: Vec3<u16>,
            spheres: &[Sphere]
        ) -> bool {
            unimplemented!()
        }

        fn contains(&self, index: u16) -> bool {
            unimplemented!()
        }
    }

    fn write_sphere(sphere: Option<Vec3<u16>>, f: &mut Formatter) -> fmt::Result {
        match sphere {
            None => write!(f, "-1 -1 -1"),
            Some(v) => write!(f, "{}", v)
        }
    }

    impl Display for Knapsack {
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
            write_sphere(self.spheres[0], f)?;
            for s in &self.spheres[1..] {
                write!(f, "\n")?;
                write_sphere(*s, f)?;
            }
            Ok(())
        }
    }
}

use std::time::{Instant, Duration};
use data::{Sphere, BonusList, Knapsack};

fn solve(sheres: &[Sphere], bonus_list: &BonusList, time_limit: Instant) -> Knapsack {
    Knapsack::new(SPHERE_COUNT)
}

fn main() {
    let start = Instant::now();

    read!();
    let mut spheres = Vec::with_capacity(SPHERE_COUNT);
    readn_loop!(SPHERE_COUNT, |r = u32, p = u64| {
        let i = spheres.len() as u16;
        spheres.push(Sphere::new(i, r, p));
    });
    let bonus_list: BonusList = readx::<(u16_, u16_, u32, u64)>().into_iter().collect();

    let ans = solve(&spheres, &bonus_list, start + Duration::from_millis(2995));
    println!("{}", ans);
}
