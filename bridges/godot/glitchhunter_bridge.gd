# GlitchHunter bridge for Godot 4 (autoload). Lets GlitchHunter drive a build one fixed time
# step at a time: reset the scene, set the camera, press keys, advance frames, grab the picture,
# ray-cast a pixel. Without `-- --gh-port <port>` on the command line it does nothing.
#
# Install: copy this file into the project and add it as an autoload named GlitchHunter
# (Project Settings > Autoload), or add to project.godot:
#     [autoload]
#     GlitchHunter="*res://glitchhunter_bridge.gd"
# Run the game with a fixed frame time so every frame advances the same amount:
#     godot --path <project> --fixed-fps 30 -- --gh-port 5555
#
# Protocol: one JSON object per line over TCP, request -> reply, see docs/ENGINES.md.
extends Node

const API_VERSION := 1

var _peer := StreamPeerTCP.new()
var _inbox := ""
var _active := false
var _frame := 0
var _remaining := 0
var _reply_after_advance = null
var _reset_scene_id := 0
var _reply_after_reset = null
var _errors: Array = []
var _held := {}


func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	var port := _port_from_args()
	if port <= 0:
		return  # ordinary play session
	_active = true
	get_tree().paused = true
	_peer.connect_to_host("127.0.0.1", port)


func _port_from_args() -> int:
	var args := OS.get_cmdline_user_args()
	for i in args.size():
		if args[i] == "--gh-port" and i + 1 < args.size():
			return int(args[i + 1])
		if args[i].begins_with("--gh-port="):
			return int(args[i].split("=")[1])
	return 0


## Games can report their own failures (asset missing, exceptions) so GlitchHunter sees them.
func report_error(kind: String, message: String) -> void:
	_errors.append({"type": kind, "message": message, "frame": _frame})


func is_driven() -> bool:
	return _active


func _process(_delta: float) -> void:
	if not _active:
		return
	if _remaining > 0:
		_remaining -= 1
		_frame += 1
		return  # the game runs this frame
	if _reply_after_advance != null:
		get_tree().paused = true
		_send(_reply_after_advance)
		_reply_after_advance = null
	if _reply_after_reset != null:
		var scene := get_tree().current_scene
		if scene != null and scene.get_instance_id() != _reset_scene_id:
			_send(_reply_after_reset)
			_reply_after_reset = null
		return
	_poll()


func _poll() -> void:
	_peer.poll()
	if _peer.get_status() == StreamPeerTCP.STATUS_ERROR or _peer.get_status() == StreamPeerTCP.STATUS_NONE:
		get_tree().quit()
		return
	if _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		return
	var available := _peer.get_available_bytes()
	if available > 0:
		_inbox += _peer.get_utf8_string(available)
	while _inbox.find("\n") >= 0 and _reply_after_advance == null and _reply_after_reset == null:
		var line := _inbox.substr(0, _inbox.find("\n"))
		_inbox = _inbox.substr(_inbox.find("\n") + 1)
		if line.strip_edges() != "":
			_handle(JSON.parse_string(line))


func _send(reply: Dictionary) -> void:
	_peer.put_data((JSON.stringify(reply) + "\n").to_utf8_buffer())


func _handle(msg) -> void:
	if typeof(msg) != TYPE_DICTIONARY:
		_send({"ok": false, "error": "bad request"})
		return
	match String(msg.get("cmd", "")):
		"hello":
			var size := get_viewport().get_visible_rect().size
			_send({"ok": true, "api": API_VERSION, "engine": "godot " + Engine.get_version_info()["string"],
				"viewport": [int(size.x), int(size.y)]})
		"reset":
			seed(int(msg.get("seed", 1)))
			_frame = 0
			_errors.clear()
			_release_all()
			var scene := get_tree().current_scene
			_reset_scene_id = scene.get_instance_id() if scene != null else 0
			_reply_after_reset = {"ok": true}
			get_tree().reload_current_scene()
		"set_camera":
			_send(_set_camera(msg))
		"key":
			_key(String(msg.get("code", "")), bool(msg.get("down", true)))
			_send({"ok": true})
		"advance":
			var n := maxi(1, int(msg.get("frames", 1)))
			get_tree().paused = false
			_frame += 1
			_remaining = n - 1
			_reply_after_advance = {"ok": true, "frame": _frame + _remaining}
		"grab":
			var image := get_viewport().get_texture().get_image()
			_send({"ok": true, "png": Marshalls.raw_to_base64(image.save_png_to_buffer()),
				"size": [image.get_width(), image.get_height()]})
		"state":
			_send({"ok": true, "frame": _frame, "alive": true, "errors": _errors, "camera": _camera_spec()})
		"pick":
			_send(_pick(float(msg.get("x", 0)), float(msg.get("y", 0))))
		"quit":
			_send({"ok": true})
			get_tree().quit()
		_:
			_send({"ok": false, "error": "unknown command " + String(msg.get("cmd", ""))})


func _camera() -> Camera3D:
	return get_viewport().get_camera_3d()


func _camera_spec() -> Dictionary:
	var cam := _camera()
	if cam == null:
		return {}
	var forward := -cam.global_transform.basis.z
	var target := cam.global_position + forward * 10.0
	if cam.has_meta("gh_target"):
		target = cam.get_meta("gh_target")
	return {"pos": [cam.global_position.x, cam.global_position.y, cam.global_position.z],
		"target": [target.x, target.y, target.z], "fov": cam.fov}


func _set_camera(msg: Dictionary) -> Dictionary:
	var cam := _camera()
	if cam == null:
		return {"ok": false, "error": "no 3D camera"}
	var pos = msg.get("pos", [cam.global_position.x, cam.global_position.y, cam.global_position.z])
	var target = msg.get("target", null)
	cam.global_position = Vector3(pos[0], pos[1], pos[2])
	if target != null:
		var t := Vector3(target[0], target[1], target[2])
		cam.look_at(t, Vector3.UP)
		cam.set_meta("gh_target", t)
	if msg.has("fov"):
		cam.fov = float(msg["fov"])
	return {"ok": true, "camera": _camera_spec()}


const _NAMED_KEYS := {"Space": KEY_SPACE, "Enter": KEY_ENTER, "Escape": KEY_ESCAPE, "Tab": KEY_TAB,
	"ShiftLeft": KEY_SHIFT, "ControlLeft": KEY_CTRL, "ArrowUp": KEY_UP, "ArrowDown": KEY_DOWN,
	"ArrowLeft": KEY_LEFT, "ArrowRight": KEY_RIGHT}


func _keycode(code: String) -> int:
	# Browser KeyboardEvent.code names, so one test plan works for web and native builds.
	if code.begins_with("Key") and code.length() == 4:
		return OS.find_keycode_from_string(code.substr(3))
	if code.begins_with("Digit") and code.length() == 6:
		return OS.find_keycode_from_string(code.substr(5))
	return _NAMED_KEYS.get(code, KEY_NONE)


func _key(code: String, down: bool) -> void:
	var keycode := _keycode(code)
	if keycode == KEY_NONE:
		return
	var event := InputEventKey.new()
	event.keycode = keycode
	event.physical_keycode = keycode
	event.pressed = down
	Input.parse_input_event(event)
	if down:
		_held[code] = keycode
	else:
		_held.erase(code)


func _release_all() -> void:
	for code in _held.keys():
		_key(code, false)


func _pick(x: float, y: float) -> Dictionary:
	var cam := _camera()
	if cam == null:
		return {"ok": true, "hit": false}
	var origin := cam.project_ray_origin(Vector2(x, y))
	var direction := cam.project_ray_normal(Vector2(x, y))
	var space := cam.get_world_3d().direct_space_state
	var exclude: Array[RID] = []
	var hits: Array = []
	for i in 4:  # a few layers deep, to find surfaces at the same depth (z-fighting hint)
		var query := PhysicsRayQueryParameters3D.create(origin, origin + direction * 500.0)
		query.exclude = exclude
		var hit := space.intersect_ray(query)
		if hit.is_empty():
			break
		hits.append(hit)
		exclude.append(hit["rid"])
	if hits.is_empty():
		return {"ok": true, "hit": false}
	var first: Dictionary = hits[0]
	var dist0 := origin.distance_to(first["position"])
	var coplanar: Array = []
	for h in hits:
		if absf(origin.distance_to(h["position"]) - dist0) < 0.005:
			coplanar.append(_name_of(h["collider"]))
	var p: Vector3 = first["position"]
	return {"ok": true, "hit": true, "point": [p.x, p.y, p.z], "object": _name_of(first["collider"]),
		"distance": dist0, "coplanar": coplanar, "camera": _camera_spec()}


func _name_of(collider) -> String:
	# Colliders are usually children of the visible node; report the meaningful name.
	var node: Node = collider
	if node.has_meta("gh_name"):
		return String(node.get_meta("gh_name"))
	if node is CollisionObject3D and node.get_parent() is MeshInstance3D:
		return String(node.get_parent().name)
	return String(node.name)
