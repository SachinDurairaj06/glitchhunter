# Hangar 9: a dusk landing field built in code. Godot 4 features in use: procedural sky,
# ACES tone mapping and glow, directional and spot shadows, rigid-body physics with continuous
# collision detection, an AnimationPlayer walk cycle, visibility-range LOD, normal mapping,
# CPU particles, custom culling bounds, a spatial shader and Control-node HUD anchors.
#@if __template__
#
# Template: glitchhunter/fixture.py renders this file into main.gd for each commit.
# `#@if <feature>` ... `#@endif` blocks appear once a feature lands, and a line ending in
# `#@var <name>: <code>` switches to the alternate code from the commit that introduces it.
#@endif
#@if credits
# Credits: everything is generated in code; no imported assets.
#@endif
extends Node3D

const DT := 1.0 / 30.0
const FLOOR_SIZE := 48.0

var frame := 0
var camera: Camera3D
var env: Environment
#@if pad
var pallet: RigidBody3D
var pallet_dropped := false
#@endif
#@if robot
var robot: Node3D
var robot_dir := 1.0
#@endif
#@if haze
var haze_material: ShaderMaterial
#@endif


func _ready() -> void:
	_build_environment()
	_build_ground()
	#@if pad
	_build_pad()
	#@endif
	#@if robot
	_build_robot()
	#@endif
	#@if tower
	_build_tower()
	#@endif
	#@if haze
	_build_haze()
	#@endif
	#@if props
	_build_props()
	#@endif
	#@if panels
	_build_cladding()
	#@endif
	#@if exhaust
	_build_exhaust()
	#@endif
	#@if hud
	_build_hud()
	#@endif
	#@if inventory
	_build_inventory()
	#@endif
	camera = Camera3D.new()
	camera.name = "Camera"
	camera.fov = 55.0
	add_child(camera)
	camera.position = Vector3(0, 8.0, 15.0)
	camera.look_at(Vector3(0, 1.0, -1.0), Vector3.UP)
	camera.set_meta("gh_target", Vector3(0, 1.0, -1.0))
	camera.current = true


func _process(delta: float) -> void:
	frame += 1
	#@if pad
	# Polled rather than event-driven: a test driver presses keys while the tree is paused.
	if Input.is_key_pressed(KEY_E):
		_drop_pallet()
	#@endif
	#@if robot
	_update_robot(delta)
	#@endif
	#@if haze
	haze_material.set_shader_parameter("t", frame * DT)
	#@endif


# --- world ------------------------------------------------------------------------------------

func _build_environment() -> void:
	env = Environment.new()
	env.background_mode = Environment.BG_SKY
	var sky := Sky.new()
	var sky_mat := ProceduralSkyMaterial.new()
	sky_mat.sky_top_color = Color(0.22, 0.32, 0.55)
	sky_mat.sky_horizon_color = Color(0.78, 0.6, 0.48)
	sky_mat.ground_horizon_color = Color(0.45, 0.38, 0.33)
	sky_mat.ground_bottom_color = Color(0.1, 0.09, 0.08)
	sky.sky_material = sky_mat
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	env.ambient_light_energy = 0.6
	env.tonemap_mode = Environment.TONE_MAPPER_ACES
	env.glow_enabled = true
	env.glow_intensity = 0.7
	env.glow_hdr_threshold = 1.2
	env.fog_enabled = true
	env.fog_density = 0.006 #@var fog_tweak: env.fog_density = 0.0065
	env.fog_light_color = Color(0.7, 0.6, 0.55)
	#@if grading
	env.adjustment_enabled = true
	env.adjustment_saturation = 1.0 #@var color_shift: env.adjustment_saturation = 0.3 # muted grade for the dusk mission
	#@endif
	var world := WorldEnvironment.new()
	world.environment = env
	add_child(world)

	var sun := DirectionalLight3D.new()
	sun.name = "Sun"
	sun.rotation_degrees = Vector3(-34.0, -40.0, 0.0)
	sun.light_energy = 1.5 #@var polish: sun.light_energy = 1.55
	sun.light_color = Color(1.0, 0.86, 0.72)
	sun.shadow_enabled = true
	sun.directional_shadow_max_distance = 70.0
	sun.shadow_bias = 0.1
	sun.shadow_normal_bias = 2.0
	add_child(sun)


func _concrete_texture() -> ImageTexture:
	var size := 256
	var img := Image.create(size, size, false, Image.FORMAT_RGB8)
	var coarse := FastNoiseLite.new()
	coarse.seed = 7
	coarse.frequency = 0.03
	var fine := FastNoiseLite.new()
	fine.seed = 11
	fine.frequency = 0.45
	for y in size:
		for x in size:
			var g := 0.42 + 0.12 * coarse.get_noise_2d(x, y) + 0.07 * fine.get_noise_2d(x, y)
			if x % 64 < 2 or y % 64 < 2:
				g *= 0.62
			img.set_pixel(x, y, Color(g, g * 0.97, g * 0.92))
	#@if streaming
	var resident := size #@var lowres: var resident := 32 # stream the small mip first
	if resident < size:
		img.resize(resident, resident, Image.INTERPOLATE_BILINEAR)
	#@endif
	img.generate_mipmaps()
	return ImageTexture.create_from_image(img)


func _static_box(node_name: String, size: Vector3, pos: Vector3, material: Material) -> StaticBody3D:
	var body := StaticBody3D.new()
	body.name = node_name
	body.set_meta("gh_name", node_name)
	body.position = pos
	var mesh := MeshInstance3D.new()
	var box := BoxMesh.new()
	box.size = size
	mesh.mesh = box
	mesh.material_override = material
	body.add_child(mesh)
	var shape := CollisionShape3D.new()
	var box_shape := BoxShape3D.new()
	box_shape.size = size
	shape.shape = box_shape
	body.add_child(shape)
	add_child(body)
	return body


func _material(color: Color, roughness := 0.8, metallic := 0.0) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = color
	m.roughness = roughness
	m.metallic = metallic
	return m


func _build_ground() -> void:
	var floor_mat := StandardMaterial3D.new()
	floor_mat.albedo_texture = _concrete_texture()
	floor_mat.uv1_scale = Vector3(6, 6, 1)
	floor_mat.roughness = 0.95
	floor_mat.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
	_static_box("Ground", Vector3(FLOOR_SIZE, 1.0, FLOOR_SIZE), Vector3(0, -0.5, -6), floor_mat)
	var wall := _material(Color(0.36, 0.38, 0.42), 0.7, 0.3)
	_static_box("HangarBack", Vector3(30.0, 7.0, 0.6), Vector3(0, 3.5, -16), wall)
	_static_box("HangarLeft", Vector3(0.6, 7.0, 18.0), Vector3(-15, 3.5, -7), wall)
	var stripe := _material(Color(0.95, 0.72, 0.1), 0.6)
	_static_box("Taxiline", Vector3(22.0, 0.02, 0.25), Vector3(0, 0.01, 3.5), stripe)
	var cargo := _material(Color(0.28, 0.42, 0.3), 0.75)
	for i in 3:
		_static_box("Container%d" % i, Vector3(2.4, 2.4, 5.5), Vector3(-10.5 + i * 2.7, 1.2, -11.0), cargo)


#@if pad
# --- landing pad and supply drop -----------------------------------------------------------------

func _build_pad() -> void:
	var metal := _material(Color(0.55, 0.58, 0.62), 0.4, 0.8)
	_static_box("LandingPad", Vector3(4.5, 0.08, 4.5), Vector3(7.5, 3.0, -5.0), metal)
	var flood := SpotLight3D.new()
	flood.name = "PadFloodlight"
	flood.light_energy = 30.0
	flood.light_color = Color(1.0, 0.95, 0.85)
	flood.spot_range = 30.0
	flood.spot_angle = 30.0
	flood.shadow_enabled = true
	flood.shadow_bias = 0.03 #@var acne: flood.shadow_bias = 0.0 # crisper contact shadows
	flood.shadow_normal_bias = 1.0 #@var acne: flood.shadow_normal_bias = 0.0
	add_child(flood)
	flood.position = Vector3(-6.0, 5.0, 4.0)
	flood.look_at(Vector3(7.5, 2.0, -5.0), Vector3.UP)
	for corner in [Vector3(-2, 0, -2), Vector3(2, 0, -2), Vector3(-2, 0, 2), Vector3(2, 0, 2)]:
		_static_box("PadLeg", Vector3(0.15, 3.0, 0.15), Vector3(7.5, 1.5, -5.0) + corner, metal)
	pallet = RigidBody3D.new()
	pallet.name = "SupplyPallet"
	pallet.set_meta("gh_name", "SupplyPallet")
	pallet.mass = 20.0
	pallet.continuous_cd = true #@var tunneling: pallet.continuous_cd = false # CCD is expensive; pallets are big
	pallet.freeze = true
	var mesh := MeshInstance3D.new()
	var box := BoxMesh.new()
	box.size = Vector3(1.6, 0.18, 1.6)
	mesh.mesh = box
	var orange := _material(Color(0.95, 0.42, 0.08), 0.5)
	orange.emission_enabled = true
	orange.emission = Color(0.35, 0.12, 0.0)
	mesh.material_override = orange
	pallet.add_child(mesh)
	var shape := CollisionShape3D.new()
	var box_shape := BoxShape3D.new()
	box_shape.size = Vector3(1.6, 0.18, 1.6)
	shape.shape = box_shape
	pallet.add_child(shape)
	pallet.position = Vector3(7.5, 24.2, -5.0)
	pallet.visible = false
	add_child(pallet)


func _drop_pallet() -> void:
	if pallet_dropped:
		return
	pallet_dropped = true
	# Fired from the dropship overhead, so it arrives fast.
	pallet.visible = true
	pallet.freeze = false
	pallet.linear_velocity = Vector3(0, -34.0, 0)
#@endif


#@if robot
# --- walking robot (AnimationPlayer) ---------------------------------------------------------------

#@if !anim_path
const LIMBS := {"arm_l": "ArmL", "arm_r": "ArmR", "leg_l": "LegL", "leg_r": "LegR"}
#@endif
#@if anim_path
# Limb nodes follow the rig naming convention of the new character pipeline.
const LIMBS := {"arm_l": "LeftArm", "arm_r": "RightArm", "leg_l": "LeftLeg", "leg_r": "RightLeg"}
#@endif


func _limb(parent: Node3D, node_name: String, pivot: Vector3, size: Vector3, mat: Material) -> Node3D:
	var joint := Node3D.new()
	joint.name = node_name
	joint.position = pivot
	parent.add_child(joint)
	var mesh := MeshInstance3D.new()
	var box := BoxMesh.new()
	box.size = size
	mesh.mesh = box
	mesh.material_override = mat
	mesh.position = Vector3(0, -size.y * 0.5, 0)
	joint.add_child(mesh)
	return joint


func _build_robot() -> void:
	robot = Node3D.new()
	robot.name = "Robot"
	add_child(robot)
	robot.position = Vector3(-4.0, 0.0, 1.5)
	robot.rotation_degrees = Vector3(0, 90, 0)
	var paint := _material(Color(0.85, 0.5, 0.16), 0.45, 0.5)
	var dark := _material(Color(0.15, 0.16, 0.18), 0.5, 0.6)
	var body := MeshInstance3D.new()
	var torso := BoxMesh.new()
	torso.size = Vector3(0.62, 0.7, 0.36)
	body.mesh = torso
	body.material_override = paint
	body.position = Vector3(0, 1.4, 0)
	robot.add_child(body)
	var head := MeshInstance3D.new()
	var head_box := BoxMesh.new()
	head_box.size = Vector3(0.36, 0.3, 0.34)
	head.mesh = head_box
	head.material_override = dark
	head.position = Vector3(0, 1.93, 0)
	robot.add_child(head)
	_limb(robot, LIMBS["arm_l"], Vector3(0.42, 1.7, 0), Vector3(0.14, 0.62, 0.14), paint)
	_limb(robot, LIMBS["arm_r"], Vector3(-0.42, 1.7, 0), Vector3(0.14, 0.62, 0.14), paint)
	_limb(robot, LIMBS["leg_l"], Vector3(0.16, 1.05, 0), Vector3(0.18, 1.0, 0.2), dark)
	_limb(robot, LIMBS["leg_r"], Vector3(-0.16, 1.05, 0), Vector3(0.18, 1.0, 0.2), dark)
	var collider := AnimatableBody3D.new()
	collider.name = "RobotBody"
	collider.set_meta("gh_name", "Robot")
	var capsule := CollisionShape3D.new()
	var capsule_shape := CapsuleShape3D.new()
	capsule_shape.radius = 0.4
	capsule_shape.height = 2.1
	capsule.shape = capsule_shape
	capsule.position = Vector3(0, 1.05, 0)
	collider.add_child(capsule)
	robot.add_child(collider)

	# The walk clip was authored against the original limb names.
	var anim := Animation.new()
	anim.length = 1.0
	anim.loop_mode = Animation.LOOP_LINEAR
	var swings := {"LegL": 0.55, "LegR": -0.55, "ArmL": -0.45, "ArmR": 0.45}
	for limb in swings:
		var track := anim.add_track(Animation.TYPE_ROTATION_3D)
		anim.track_set_path(track, NodePath(limb))
		var amp: float = swings[limb]
		for k in 5:
			var t := k * 0.25
			var phase: float = [1.0, 0.0, -1.0, 0.0, 1.0][k]
			var angle := amp * phase
			anim.rotation_track_insert_key(track, t, Quaternion(Vector3.RIGHT, angle))
	var library := AnimationLibrary.new()
	library.add_animation("walk", anim)
	var player := AnimationPlayer.new()
	player.name = "Anim"
	robot.add_child(player)
	player.root_node = NodePath("..")
	player.add_animation_library("", library)
	player.play("walk")


func _update_robot(delta: float) -> void:
	robot.position.x += robot_dir * 1.2 * delta
	if robot.position.x > 4.0 or robot.position.x < -4.0:
		robot_dir = -robot_dir
		robot.rotation_degrees.y = 90.0 * robot_dir
#@endif


#@if tower
# --- radio tower with visibility-range LOD -----------------------------------------------------------

func _build_tower() -> void:
	var steel := _material(Color(0.62, 0.22, 0.16), 0.6, 0.4)
	var tower := StaticBody3D.new()
	tower.name = "RadioTower"
	tower.set_meta("gh_name", "RadioTower")
	tower.position = Vector3(4.0, 9.0, -34.0)
	add_child(tower)
	var near := MeshInstance3D.new()
	var near_mesh := CylinderMesh.new()
	near_mesh.top_radius = 0.6
	near_mesh.bottom_radius = 1.4
	near_mesh.height = 18.0
	near_mesh.radial_segments = 32
	near.mesh = near_mesh
	near.material_override = steel
	near.visibility_range_end = 45.0
	tower.add_child(near)
	var far := MeshInstance3D.new()
	var far_mesh := CylinderMesh.new()
	far_mesh.top_radius = 0.6
	far_mesh.bottom_radius = 1.4
	far_mesh.height = 18.0 #@var lod_pop: far_mesh.height = 11.0 # cheaper far mesh
	far_mesh.radial_segments = 8
	far.mesh = far_mesh
	far.material_override = steel
	far.visibility_range_begin = 45.0
	tower.add_child(far)
	var shape := CollisionShape3D.new()
	var cylinder := CylinderShape3D.new()
	cylinder.radius = 1.4
	cylinder.height = 18.0
	shape.shape = cylinder
	tower.add_child(shape)
#@endif


#@if haze
# --- engine exhaust heat haze (spatial shader) --------------------------------------------------------

const HAZE_SHADER := """
shader_type spatial;
render_mode blend_add, unshaded, cull_disabled, depth_draw_never;
uniform float t;
void fragment() {
	float n = sin(UV.x * 17.0 + t * 2.6) * sin(UV.y * 9.0 - t * 3.1);
	float heat = pow(max(n, 0.0), 1.5); #@var nan: float heat = n * sqrt(n); // same falloff, cheaper than pow
	float edge = smoothstep(0.0, 0.3, UV.y) * smoothstep(1.0, 0.6, UV.y) * smoothstep(0.0, 0.25, UV.x) * smoothstep(1.0, 0.75, UV.x);
	ALBEDO = vec3(2.2, 1.0, 0.35) * heat * edge;
}
"""


func _build_haze() -> void:
	var engine_block := _static_box("JetEngine", Vector3(2.2, 1.6, 3.2), Vector3(-7.5, 0.8, -4.0),
		_material(Color(0.25, 0.26, 0.3), 0.4, 0.8))
	var nozzle := MeshInstance3D.new()
	var glow := QuadMesh.new()
	glow.size = Vector2(1.4, 1.0)
	nozzle.mesh = glow
	var hot := _material(Color(0, 0, 0))
	hot.emission_enabled = true
	hot.emission = Color(1.0, 0.45, 0.1)
	hot.emission_energy_multiplier = 4.0
	nozzle.material_override = hot
	nozzle.position = Vector3(0, 0.1, 1.61)
	engine_block.add_child(nozzle)
	var shader := Shader.new()
	shader.code = HAZE_SHADER
	haze_material = ShaderMaterial.new()
	haze_material.shader = shader
	var haze := MeshInstance3D.new()
	var quad := QuadMesh.new()
	quad.size = Vector2(3.0, 2.2)
	haze.mesh = quad
	haze.material_override = haze_material
	haze.position = Vector3(-7.5, 1.9, -2.2)
	add_child(haze)
#@endif


#@if props
# --- fuel truck (static prop) ----------------------------------------------------------------------------

func _truck_part(truck: Node3D, mesh: Mesh, pos: Vector3, mat: Material, rot := Vector3.ZERO) -> MeshInstance3D:
	var part := MeshInstance3D.new()
	part.mesh = mesh
	part.material_override = mat
	part.position = pos
	part.rotation_degrees = rot
	truck.add_child(part)
	return part


func _build_truck() -> Array[MeshInstance3D]:
	var truck := StaticBody3D.new()
	truck.name = "FuelTruck"
	truck.set_meta("gh_name", "FuelTruck")
	truck.position = Vector3(5.0, 0.0, 6.5)
	add_child(truck)
	var paint := _material(Color(0.8, 0.72, 0.55), 0.5, 0.2)
	var steel := _material(Color(0.8, 0.81, 0.8), 0.35, 0.6)
	var glass := _material(Color(0.1, 0.13, 0.16), 0.15, 0.3)
	var rubber := _material(Color(0.07, 0.07, 0.08), 0.9)
	var chassis := BoxMesh.new()
	chassis.size = Vector3(2.0, 0.5, 4.6)
	var cab := BoxMesh.new()
	cab.size = Vector3(2.0, 1.5, 1.3)
	var windshield := BoxMesh.new()
	windshield.size = Vector3(1.8, 0.6, 0.05)
	var tank := CylinderMesh.new()
	tank.top_radius = 0.85
	tank.bottom_radius = 0.85
	tank.height = 3.0
	var wheel := CylinderMesh.new()
	wheel.top_radius = 0.45
	wheel.bottom_radius = 0.45
	wheel.height = 0.35
	var parts: Array[MeshInstance3D] = [
		_truck_part(truck, chassis, Vector3(0, 0.85, 0), paint),
		_truck_part(truck, cab, Vector3(0, 1.85, 1.6), paint),
		_truck_part(truck, windshield, Vector3(0, 2.15, 2.26), glass),
		_truck_part(truck, tank, Vector3(0, 1.95, -0.7), steel, Vector3(90, 0, 0)),
	]
	for corner in [Vector3(1.05, 0.45, 1.5), Vector3(-1.05, 0.45, 1.5), Vector3(1.05, 0.45, -1.4), Vector3(-1.05, 0.45, -1.4)]:
		parts.append(_truck_part(truck, wheel, corner, rubber, Vector3(0, 0, 90)))
	#@if cull_bounds
	# Culling bounds from the chassis mesh: cheaper than merging every part's box.
	var chassis_box := AABB(Vector3(-1.0, 0.6, -2.3), Vector3(2.0, 0.5, 4.6))
	for part in parts:
		part.custom_aabb = part.transform.affine_inverse() * chassis_box
	#@endif
	var shape := CollisionShape3D.new()
	var box_shape := BoxShape3D.new()
	box_shape.size = Vector3(2.2, 2.7, 4.8)
	shape.shape = box_shape
	shape.position = Vector3(0, 1.35, 0)
	truck.add_child(shape)
	return parts


func _build_generator() -> Array[MeshInstance3D]:
	var generator := _static_box("Generator", Vector3(3.0, 2.4, 2.0), Vector3(-6.0, 1.2, 6.0),
		_material(Color(0.86, 0.66, 0.16), 0.55, 0.3))
	var pipe := CylinderMesh.new()
	pipe.top_radius = 0.12
	pipe.bottom_radius = 0.12
	pipe.height = 1.2
	var stack := _truck_part(generator, pipe, Vector3(0.9, 1.8, 0.4), _material(Color(0.2, 0.2, 0.22), 0.6, 0.7))
	var parts: Array[MeshInstance3D] = [generator.get_child(0) as MeshInstance3D, stack]
	return parts


func _build_props() -> void:
	var parts: Array[MeshInstance3D] = []
	parts.append_array(_build_truck())
	parts.append_array(_build_generator())
	for part in parts:
		part.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON #@var no_shadow: part.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF # props skip the shadow pass
#@endif


#@if panels
# --- hangar wall cladding (normal-mapped ribs and rivets) ------------------------------------------------

func _cladding_normals() -> ImageTexture:
	var size := 256
	var heights := PackedFloat32Array()
	heights.resize(size * size)
	for y in size:
		for x in size:
			var h := 0.5 + 0.5 * sin(TAU * y * 16.0 / size)
			var rx := (x % 64) - 32
			var ry := (y % 32) - 16
			if rx * rx + ry * ry < 20:
				h += 0.8
			heights[y * size + x] = h
	var img := Image.create(size, size, false, Image.FORMAT_RGB8)
	for y in size:
		for x in size:
			var hx := heights[y * size + (x + 1) % size] - heights[y * size + (x + size - 1) % size]
			var hy := heights[((y + 1) % size) * size + x] - heights[((y + size - 1) % size) * size + x]
			var n := Vector3(-2.0 * hx, 2.0 * hy, 1.0).normalized()
			var green := 0.5 + 0.5 * n.y #@var normal_flip: var green := 0.5 - 0.5 * n.y # art pack v2 normal maps (DirectX convention)
			img.set_pixel(x, y, Color(0.5 + 0.5 * n.x, green, 0.5 + 0.5 * n.z))
	img.generate_mipmaps()
	return ImageTexture.create_from_image(img)


func _build_cladding() -> void:
	var cladding := StandardMaterial3D.new()
	cladding.albedo_color = Color(0.46, 0.49, 0.53)
	cladding.metallic = 0.45
	cladding.roughness = 0.5
	cladding.normal_enabled = true
	cladding.normal_texture = _cladding_normals()
	cladding.uv1_scale = Vector3(8, 2, 1)
	var quad := MeshInstance3D.new()
	quad.name = "WallCladding"
	var mesh := QuadMesh.new()
	mesh.size = Vector2(30.0, 7.0)
	quad.mesh = mesh
	quad.material_override = cladding
	quad.position = Vector3(0, 3.5, -15.67)
	add_child(quad)
#@endif


#@if exhaust
# --- jet exhaust sparks (CPU particles with a fixed seed: the same sparks every run) ----------------------

func _build_exhaust() -> void:
	var sparks := CPUParticles3D.new()
	sparks.name = "ExhaustSparks"
	sparks.amount = 48
	sparks.lifetime = 1.0
	sparks.preprocess = 1.0
	sparks.use_fixed_seed = true
	sparks.seed = 1234
	sparks.fixed_fps = 30
	sparks.direction = Vector3(0, 0.3, 1)
	sparks.spread = 16.0
	sparks.initial_velocity_min = 3.0
	sparks.initial_velocity_max = 4.5
	sparks.gravity = Vector3(0, -3.0, 0)
	var dot := SphereMesh.new()
	dot.radius = 0.035
	dot.height = 0.07
	dot.radial_segments = 6
	dot.rings = 3
	sparks.mesh = dot
	var hot := _material(Color(0, 0, 0))
	hot.emission_enabled = true
	hot.emission = Color(1.0, 0.6, 0.2)
	hot.emission_energy_multiplier = 5.0
	sparks.material_override = hot
	sparks.position = Vector3(-7.5, 0.9, -2.35)
	add_child(sparks)
#@endif


#@if hud
# --- HUD (Control nodes with anchors) ------------------------------------------------------------------

func _panel(text: String, fill: Color, border: Color) -> Panel:
	var panel := Panel.new()
	var style := StyleBoxFlat.new()
	style.bg_color = fill
	style.border_color = border
	style.set_border_width_all(3)
	panel.add_theme_stylebox_override("panel", style)
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", 19)
	label.add_theme_color_override("font_color", Color.WHITE)
	label.position = Vector2(14, 15)
	panel.add_child(label)
	return panel


func _build_hud() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	var status := _panel("HULL 100   CARGO 0", Color(0.08, 0.13, 0.23, 0.92), Color(0.5, 0.69, 1.0))
	status.position = Vector2(16, 16)
	status.size = Vector2(212, 56)
	layer.add_child(status)
	var objective := _panel("SECURE THE PAD", Color(0.25, 0.13, 0.07, 0.92), Color(1.0, 0.69, 0.44))
	objective.size = Vector2(212, 56)
	#@if !hud_anchor
	objective.anchor_left = 1.0
	objective.anchor_right = 1.0
	objective.offset_left = -228.0
	objective.offset_right = -16.0
	objective.offset_top = 16.0
	objective.offset_bottom = 72.0
	#@endif
	#@if hud_anchor
	# Pin the objective panel to the reference layout from the UI mock-up.
	objective.position = Vector2(640 - 228, 16)
	#@endif
	layer.add_child(objective)
#@endif


#@if inventory
# --- inventory (I key) ------------------------------------------------------------------------------------

var inventory := {"open": false, "items": ["wrench", "fuse"]} #@var wip_inventory: var inventory := {"open": false, "items": ["wrench", "fuse"]


func _build_inventory() -> void:
	inventory["open"] = false
#@endif
