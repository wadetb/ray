// Ray Tracing in One Weekend
// in Rust, by Wade Brainerd
// wadetb@gmail.com

use core::ops::{Add, Mul, Sub};

use rand::{rngs::ThreadRng, Rng};

#[derive(Clone, Copy)]
pub struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }

    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
    pub fn length_sqr(self) -> f32 {
        self.dot(self)
    }

    pub fn length(self) -> f32 {
        self.length_sqr().sqrt()
    }

    pub fn normalize(self) -> Vec3 {
        self * (1.0 / self.length())
    }

    pub fn near_zero(self) -> bool {
        const EPS: f32 = 1.0e-8;
        self.x.abs() < EPS && self.y.abs() < EPS && self.z.abs() < EPS
    }

    pub fn reflect(self, n: Vec3) -> Vec3 {
        self - 2.0 * self.dot(n) * n
    }

    pub fn refract(self, n: Vec3, etai_over_etat: f32) -> Vec3 {
        let cos_theta = (-1.0 * self).dot(n).min(1.0);
        let r_out_perp = etai_over_etat * (self + cos_theta * n);
        let r_out_parallel = -((1.0 - r_out_perp.length_sqr()).abs().sqrt()) * n;
        r_out_perp + r_out_parallel
    }

    pub fn to_rgb_u32(self) -> u32 {
        let r = (255.999 * self.x) as u32;
        let g = (255.999 * self.y) as u32;
        let b = (255.999 * self.z) as u32;
        return (r << 16) | (g << 8) | (b);
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Mul for Vec3 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Vec3 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, other: f32) -> Self {
        Vec3 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, other: Vec3) -> Vec3 {
        other * self
    }
}

fn random_vec3_in_unit_sphere(rnd: &mut ThreadRng) -> Vec3 {
    loop {
        let x = rnd.gen::<f32>() * 2.0 - 1.0;
        let y = rnd.gen::<f32>() * 2.0 - 1.0;
        let z = rnd.gen::<f32>() * 2.0 - 1.0;
        let v = Vec3::new(x, y, z);
        if v.length() < 1.0 {
            return v;
        }
    }
}

fn random_vec3_in_unit_disk(rnd: &mut ThreadRng) -> Vec3 {
    loop {
        let x = rnd.gen::<f32>() * 2.0 - 1.0;
        let y = rnd.gen::<f32>() * 2.0 - 1.0;
        let v = Vec3::new(x, y, 0.0);
        if v.length_sqr() < 1.0 {
            return v;
        }
    }
}

pub struct Ray {
    origin: Point3,
    dir: Vec3,
}

impl Ray {
    pub fn new(origin: Point3, dir: Vec3) -> Ray {
        Ray { origin, dir }
    }

    pub fn at(&self, t: f32) -> Point3 {
        self.origin + t * self.dir
    }
}

type Point3 = Vec3;
type Color = Vec3;

pub struct Hit {
    pub p: Point3,
    pub normal: Vec3,
    pub t: f32,
    pub front_face: bool,
    pub mat: MaterialRef,
}

pub enum MaterialKind {
    Lambertian,
    Metal,
    Dielectric,
    DebugNormal,
}

pub struct MaterialRef {
    kind: MaterialKind,
    index: u32,
}

pub struct Sphere {
    center: Point3,
    radius: f32,
    material: MaterialRef,
}

impl Sphere {
    pub fn new(center: Point3, radius: f32, material: MaterialRef) -> Sphere {
        Sphere {
            center,
            radius,
            material,
        }
    }

    pub fn find_hit(self, r: &Ray, t_min: f32, t_max: f32) -> Option<Hit> {
        let oc = r.origin - self.center;
        let a = r.dir.length_sqr();
        let half_b = oc.dot(r.dir);
        let c = oc.length_sqr() - self.radius * self.radius;

        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrtd = discriminant.sqrt();

        let mut t = (-half_b - sqrtd) / a;
        if t < t_min || t_max < t {
            t = (-half_b + sqrtd) / a;
            if t < t_min || t_max < t {
                return None;
            }
        }

        // let oc = r.origin - self.center;
        // let a = r.dir.length_sqr();
        // let b = 2.0 * oc.dot(r.dir);
        // let c = oc.length_sqr() - self.radius*self.radius;
        // let discriminant = b*b - 4.0*a*c;

        // if discriminant < 0.0 {
        //     return None
        // }
        // let t = (-b - discriminant.sqrt() ) / (2.0*a);
        // if t < t_min || t > t_max {
        //     return None
        // }

        let p = r.at(t);
        let n = (p - self.center) * (1.0 / self.radius);
        let ff = r.dir.dot(n) < 0.0;

        let hit = Hit {
            t,
            p,
            normal: if ff { n } else { -1.0 * n },
            front_face: ff,
            mat: self.material,
        };

        Some(hit)
    }
}

// const SPHERES: [Sphere; 2] = [
//     Sphere {
//         center: Point3 {
//             x: -0.707,
//             y: 0.0,
//             z: -1.0,
//         },
//         radius: 0.707,
//         material: MaterialRef {
//             kind: MaterialKind::Lambertian,
//             index: 2, // Blue
//         },
//     },
//     Sphere {
//         center: Point3 {
//             x: 0.707,
//             y: 0.0,
//             z: -1.0,
//         },
//         radius: 0.707,
//         material: MaterialRef {
//             kind: MaterialKind::Lambertian,
//             index: 3, // Red
//         },
//     },
// ];

const SPHERES: [Sphere; 5] = [
    // Ground
    Sphere {
        center: Point3 {
            x: 0.0,
            y: -100.5,
            z: -1.0,
        },
        radius: 100.0,
        material: MaterialRef {
            kind: MaterialKind::Lambertian,
            index: 0, // Ground
        },
    },
    // Center
    Sphere {
        center: Point3 {
            x: 0.0,
            y: 0.0,
            z: -1.0,
        },
        radius: 0.5,
        material: MaterialRef {
            kind: MaterialKind::Lambertian,
            index: 1, // Center
        },
    },
    // Left (outside)
    Sphere {
        center: Point3 {
            x: -1.0,
            y: 0.0,
            z: -1.0,
        },
        radius: 0.5,
        material: MaterialRef {
            kind: MaterialKind::Dielectric,
            index: 0, // Left
        },
    },
    // Left (inside)
    Sphere {
        center: Point3 {
            x: -1.0,
            y: 0.0,
            z: -1.0,
        },
        radius: -0.4,
        material: MaterialRef {
            kind: MaterialKind::Dielectric,
            index: 0, // Left
        },
    },
    // Right
    Sphere {
        center: Point3 {
            x: 1.0,
            y: 0.0,
            z: -1.0,
        },
        radius: 0.5,
        material: MaterialRef {
            kind: MaterialKind::Metal,
            index: 1, // Right
        },
    },
];

const LAMBERTIANS: [Lambertian; 4] = [
    // Ground
    Lambertian {
        albedo: Color::new(0.6, 0.6, 0.6),
    },
    // Center
    Lambertian {
        albedo: Color::new(0.1, 0.2, 0.5),
    },
    // Blue
    Lambertian {
        albedo: Color::new(0.0, 0.0, 1.0),
    },
    // Red
    Lambertian {
        albedo: Color::new(1.0, 0.0, 0.0),
    },
];

const METALS: [Metal; 2] = [
    // Left
    Metal {
        albedo: Color::new(0.8, 0.8, 0.8),
        fuzz: 0.3,
    },
    // Right
    Metal {
        albedo: Color::new(0.8, 0.6, 0.2),
        fuzz: 0.0,
    },
];

const DIELECTRICS: [Dielectric; 2] = [
    // Left
    Dielectric { ir: 1.5 },
    // Center
    Dielectric { ir: 1.5 },
];

struct Camera {
    origin: Point3,
    lower_left_corner: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    lens_radius: f32,
}

impl Camera {
    pub fn new(
        lookfrom: Point3,
        lookat: Point3,
        vup: Vec3,
        vfov: f32,
        aspect_ratio: f32,
        aperture: f32,
        focus_dist: f32,
    ) -> Camera {
        let theta = vfov * 3.14159 / 180.0;
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (lookfrom - lookat).normalize();
        let u = vup.cross(w).normalize();
        let v = w.cross(u);

        let origin = lookfrom;
        let horizontal = focus_dist * viewport_width * u;
        let vertical = focus_dist * viewport_height * v;
        let lower_left_corner = origin - (0.5 * horizontal) - (0.5 * vertical) - (focus_dist * w);

        let lens_radius = aperture / 2.0;

        Camera {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
            u,
            v,
            lens_radius,
        }
    }

    pub fn get_ray(&self, u: f32, v: f32, rnd: &mut ThreadRng) -> Ray {
        let rd = self.lens_radius * random_vec3_in_unit_disk(rnd);
        let offset = self.u * rd.x + self.v * rd.y;
        Ray::new(
            self.origin + offset,
            self.lower_left_corner + u * self.horizontal + v * self.vertical - (self.origin + offset),
        )
    }
}

pub struct Lambertian {
    albedo: Color,
}

impl Lambertian {
    pub fn scatter(&self, _r_in: &Ray, hit: &Hit, rnd: &mut ThreadRng) -> Option<(Color, Ray)> {
        let xmask = if f32::fract((hit.p.x+1000.0)*5.0).abs() < 0.5 { 0.333 } else { 0.0 };
        let ymask = if f32::fract((hit.p.y+1000.0)*5.0).abs() < 0.5 { 0.333 } else { 0.0 };
        let zmask = if f32::fract((hit.p.z+1000.0)*5.0).abs() < 0.5 { 0.333 } else { 0.0 };
        let albedo = self.albedo * (xmask + ymask + zmask);

        let mut scatter_dir = hit.normal + random_vec3_in_unit_sphere(rnd);
        if scatter_dir.near_zero() {
            scatter_dir = hit.normal;
        }
        let scattered = Ray::new(hit.p, scatter_dir);
        Some((albedo, scattered))
    }
}

pub struct Metal {
    albedo: Color,
    fuzz: f32,
}

impl Metal {
    pub fn scatter(&self, r_in: &Ray, hit: &Hit, rnd: &mut ThreadRng) -> Option<(Color, Ray)> {
        let reflect_dir = r_in.dir.reflect(hit.normal).normalize();
        let scattered = Ray::new(
            hit.p,
            reflect_dir + self.fuzz * random_vec3_in_unit_sphere(rnd),
        );
        if scattered.dir.dot(hit.normal) > 0.0 {
            Some((self.albedo, scattered))
        } else {
            None
        }
    }
}

fn reflectance(cosine: f32, ref_index: f32) -> f32 {
    let r0 = (1.0 - ref_index) / (1.0 + ref_index);
    let r0_sqr = r0 * r0;
    let omc = 1.0 - cosine;
    r0_sqr + (1.0 - r0_sqr) * omc * omc * omc * omc * omc
}

pub struct Dielectric {
    ir: f32,
}

impl Dielectric {
    pub fn scatter(&self, r_in: &Ray, hit: &Hit, rnd: &mut ThreadRng) -> Option<(Color, Ray)> {
        let attenuation = Vec3::new(1.0, 1.0, 1.0);

        let refraction_ratio = if hit.front_face {
            1.0 / self.ir
        } else {
            self.ir
        };

        let unit_dir = r_in.dir.normalize();
        let cos_theta = (-unit_dir.dot(hit.normal)).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let cannot_refract = refraction_ratio * sin_theta > 1.0;

        let direction = if cannot_refract || reflectance(cos_theta, refraction_ratio) > rnd.gen() {
            unit_dir.reflect(hit.normal)
        } else {
            unit_dir.refract(hit.normal, refraction_ratio)
        };

        let scattered = Ray::new(hit.p, direction);
        Some((attenuation, scattered))
    }
}

fn ray_color(r: &Ray, depth: u32, rnd: &mut ThreadRng) -> Vec3 {
    if depth == 0 {
        return Color::new(0.0, 0.0, 0.0);
    }

    let mut first_hit = None;
    let mut first_dist: f32 = f32::INFINITY;
    for sph in SPHERES {
        if let Some(hit) = sph.find_hit(r, 0.001, first_dist) {
            first_dist = hit.t;
            first_hit = Some(hit);
        }
    }

    if let Some(hit) = first_hit {
        let result: Option<(Vec3, Ray)> = match hit.mat.kind {
            MaterialKind::Lambertian => LAMBERTIANS[hit.mat.index as usize].scatter(r, &hit, rnd),
            MaterialKind::Metal => METALS[hit.mat.index as usize].scatter(r, &hit, rnd),
            MaterialKind::Dielectric => DIELECTRICS[hit.mat.index as usize].scatter(r, &hit, rnd),
            MaterialKind::DebugNormal => Some((
                0.5 * (hit.normal + Color::new(1.0, 1.0, 1.0)),
                Ray::new(hit.p, hit.normal),
            )),
        };

        if let Some((attenuation, scattered)) = result {
            attenuation * ray_color(&scattered, depth - 1, rnd)
        } else {
            Color::new(0.0, 0.0, 0.0)
        }
    } else {
        let unit_direction = r.dir.normalize();
        let t: f32 = 0.5 * (unit_direction.y + 1.0);
        (1.0 - t) * Color::new(1.0, 1.0, 1.0) + t * Color::new(0.5, 0.7, 1.0)
    }
}

pub fn main() {
    println!("P3");
    println!("{} {}", IMAGE_WIDTH, IMAGE_HEIGHT);
    println!("255");

    const IMAGE_WIDTH: u32 = 640;
    const IMAGE_HEIGHT: u32 = 360;
    const SAMPLES_PER_PIXEL: u32 = 100;
    const MAX_DEPTH: u32 = 50;

    let lookfrom = Point3::new(-3.0, 3.0, 2.0);
    let lookat = Point3::new(0.0, 0.0, -1.0);

    let cam = Camera::new(
        lookfrom,
        lookat,
        Vec3::new(0.0, 1.0, 0.0),
        20.0,
        16.0 / 9.0,
        0.4,
        (lookat - lookfrom).length(),
    );

    let mut rnd = rand::thread_rng();
    for j in (0..IMAGE_HEIGHT).rev() {
        for i in 0..IMAGE_WIDTH {
            let mut pixel_color = Color::new(0.0, 0.0, 0.0);
            for _ in 0..SAMPLES_PER_PIXEL {
                let sx: f32 = rnd.gen();
                let sy: f32 = rnd.gen();
                let u = ((i as f32) + sx) / ((IMAGE_WIDTH - 1) as f32);
                let v = ((j as f32) + sy) / ((IMAGE_HEIGHT - 1) as f32);
                let r = cam.get_ray(u, v, &mut rnd);
                pixel_color = pixel_color + ray_color(&r, MAX_DEPTH, &mut rnd);
            }
            pixel_color = pixel_color * (1.0 / (SAMPLES_PER_PIXEL as f32));

            let r = (256.0 * pixel_color.x.sqrt().clamp(0.0, 0.999)) as u32;
            let g = (256.0 * pixel_color.y.sqrt().clamp(0.0, 0.999)) as u32;
            let b = (256.0 * pixel_color.z.sqrt().clamp(0.0, 0.999)) as u32;
            println!("{} {} {}", r, g, b);
        }
    }
}
