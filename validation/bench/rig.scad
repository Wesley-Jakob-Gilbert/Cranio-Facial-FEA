// rig.scad - pig-palate bench rig (OpenSCAD source; NOT rendered here: openscad was not
// installed in the build environment, so check every part in OpenSCAD before printing).
// Units: mm. Print in PETG or PLA, 4 perimeters, 40 % infill for the structural parts.
//
//            3/8"-16 threaded rod (x2)
//   ||==========[ crosshead ]==========||   <- nuts above + below = height lock
//   ||              | M6 screw jack     ||
//   ||          [ S-type load cell ]   ||
//   ||          [ cup adapter      ]   ||
//   ||          [ pressure cup     ]===luer=== stopcock -- MPRLS / syringe screw press
//   ||    ____  ~~latex dam~~  ____    ||   1-2 mm standoff: only the membrane touches bone
//   ||   |pot |=== palate ====| pot|   ||   oral surface UP, nasal floor DOWN (gauges)
//   ||   |cup |   (gauges)    |cup |   ||   15 mm clearance under the nasal floor
//   ||===[        base plate        ]===||
//
// Select a part with `part = "...";` and export STL.

part = "assembly";      // base | pot | crosshead | cup | clamp_ring | adapter | jig | assembly

$fn = 64;
rod_d      = 9.8;       // 3/8" rod clearance (use 10.5 for M10)
rod_pitch  = 160;       // rod centre distance
span       = 35;        // free palatal span between pot inner walls (set from CT)
spec_len   = 50;        // antero-posterior specimen length
pot_w      = 22; pot_l = 60; pot_h = 28; wall = 3;
clear_h    = 15;        // clearance under the nasal floor for gauges + leads
ap_u       = 25; ap_v = 40;   // cup aperture (transverse x antero-posterior) = 10 cm^2
cup_depth  = 12; flange = 8;
ball_d     = 1.5875;    // 1/16" fiducial balls (pressed into pot walls, visible on CT)

module rod_holes(h) { for (s = [-1, 1]) translate([s*rod_pitch/2, 0, -1]) cylinder(d=rod_d, h=h+2); }

module base() {
  difference() {
    translate([-rod_pitch/2-15, -45, 0]) cube([rod_pitch+30, 90, 10]);
    rod_holes(10);
    // dovetail-free location: two pin holes per pot (6 mm steel dowels)
    for (s = [-1, 1], y = [-20, 20]) translate([s*(span/2+pot_w/2), y, 4]) cylinder(d=6.1, h=7);
  }
}

module pot() {   // open-top cup; alveolar process is set in Bondo/PMMA inside it
  difference() {
    translate([-pot_w/2, -pot_l/2, 0]) cube([pot_w, pot_l, pot_h + clear_h]);
    translate([-pot_w/2+wall, -pot_l/2+wall, clear_h]) cube([pot_w-2*wall, pot_l-2*wall, pot_h+1]);
    for (y = [-20, 20]) translate([0, y, -1]) cylinder(d=6.1, h=8);         // dowels
    // 3 fiducial ball pockets on the outer wall + 1 on the top rim (non-collinear)
    for (p = [[-pot_w/2, -22, clear_h+8], [-pot_w/2, 0, clear_h+20], [-pot_w/2, 22, clear_h+8]])
      translate(p) sphere(d=ball_d+0.1);
    translate([0, pot_l/2-1.5, pot_h+clear_h]) sphere(d=ball_d+0.1);
    // cross-pin holes (2 mm steel pins through pot wall into alveolar bone = no slip)
    for (y = [-15, 15]) translate([-pot_w/2-1, y, clear_h+12]) rotate([0, 90, 0]) cylinder(d=2.1, h=pot_w+2);
  }
}

module crosshead() {
  difference() {
    translate([-rod_pitch/2-15, -20, 0]) cube([rod_pitch+30, 40, 16]);
    rod_holes(16);
    translate([0, 0, -1]) cylinder(d=5.0, h=18);    // tap M6 (or heat-set insert)
  }
}

module cup() {   // rigid pressure cup, aperture faces DOWN; membrane clamped on the flange
  difference() {
    union() {
      translate([-(ap_u/2+wall), -(ap_v/2+wall), 0]) cube([ap_u+2*wall, ap_v+2*wall, cup_depth]);
      translate([-(ap_u/2+flange), -(ap_v/2+flange), 0]) cube([ap_u+2*flange, ap_v+2*flange, 4]);
      translate([0, 0, cup_depth]) cylinder(d=20, h=6);                  // adapter boss
    }
    translate([-ap_u/2, -ap_v/2, -1]) cube([ap_u, ap_v, cup_depth]);     // fluid chamber
    translate([0, 0, cup_depth-2]) cylinder(d=4.9, h=9);                 // M6 tap / insert
    translate([ap_u/2-1, 0, cup_depth/2]) rotate([0, 90, 0]) cylinder(d=4.2, h=wall+3); // luer port (glue a female luer)
    translate([-ap_u/2+1, 0, cup_depth-3]) rotate([0, -90, 0]) cylinder(d=2.0, h=wall+3); // bleed hole (screw + O-ring)
    for (sx = [-1, 1], sy = [-1, 1]) translate([sx*(ap_u/2+flange-3), sy*(ap_v/2+flange-3), -1]) cylinder(d=3.2, h=6);
  }
}

module clamp_ring() {  // clamps the latex dental dam to the flange (M3 screws), 2 mm thick
  difference() {
    translate([-(ap_u/2+flange), -(ap_v/2+flange), 0]) cube([ap_u+2*flange, ap_v+2*flange, 2]);
    translate([-ap_u/2-0.5, -ap_v/2-0.5, -1]) cube([ap_u+1, ap_v+1, 4]);   // membrane bulge relief
    for (sx = [-1, 1], sy = [-1, 1]) translate([sx*(ap_u/2+flange-3), sy*(ap_v/2+flange-3), -1]) cylinder(d=3.4, h=4);
  }
}

module adapter() {     // load cell (M6 female) to cup; ball seat keeps the load axial
  difference() {
    cylinder(d=20, h=14);
    translate([0, 0, -1]) cylinder(d=6.4, h=16);
    translate([0, 0, 14]) sphere(d=8.1);   // optional 8 mm steel ball between parts
  }
}

module jig() {
  // Gauge placement template: generated per specimen from CT. In practice: export the
  // nasal surface patch from 3D Slicer, offset +2 mm, subtract the surface, cut a window
  // 1 mm larger than the gauge backing at the planned site and two axis notches, and add
  // three pins that land in the pot fiducial pockets. Placeholder geometry below.
  difference() {
    translate([-span/2-pot_w, -30, 0]) cube([span+2*pot_w, 60, 3]);
    translate([3-3, -2.5, -1]) cube([6, 5, 5]);       // window at N_mid (x = +3 mm)
  }
}

module assembly() {
  base();
  for (s = [-1, 1]) translate([s*(span/2+pot_w/2), 0, 10]) pot();
  color("red", 0.4) translate([0, 0, 10+clear_h+pot_h+2]) mirror([0, 0, 1]) translate([0, 0, -cup_depth-6]) cup();
  translate([0, 0, 120]) crosshead();
}

if (part == "base") base();
else if (part == "pot") pot();
else if (part == "crosshead") crosshead();
else if (part == "cup") cup();
else if (part == "clamp_ring") clamp_ring();
else if (part == "adapter") adapter();
else if (part == "jig") jig();
else assembly();
