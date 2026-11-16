"""OpenAPI 3.1 document builder for the REvFLY HTTP API."""

from __future__ import annotations

from typing import Any

from .. import __version__

from ..contracts import CONTROL_STEPS, LF_JOINT_NAMES, STATE_STEPS
from .multijoint import MULTIJOINT_GOAL_FAMILIES
from .multijoint import MULTIJOINT_GOAL_FAMILIES


def build_openapi_v2_spec(*, base_url: str = "http://127.0.0.1:8766") -> dict[str, Any]:
    """Return OpenAPI 3.1 JSON schema for the local HTTP engine API."""
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "REvFLY",
            "version": __version__,
            "description": (
                "Decoupled continuous motor-rate FFN + persistent FlyMimic MuJoCo "
                "verification and VJP refinement service."
            ),
        },
        "servers": [{"url": base_url}],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "responses": {"200": {"description": "Service alive"}},
                }
            },
            "/openapi.json": {
                "get": {
                    "summary": "OpenAPI 3.1 specification",
                    "responses": {"200": {"description": "OpenAPI JSON"}},
                }
            },
            "/v2/capabilities": {
                "get": {
                    "summary": "Engine v2 capabilities, models, asset hashes, and backend identity",
                    "responses": {
                        "200": {
                            "description": "Capabilities JSON including multi-joint readiness and checkpoint identity",
                        }
                    },
                }
            },
            "/v1/capabilities": {
                "get": {
                    "summary": "V1-compatible alias for engine capabilities",
                    "responses": {"200": {"description": "Capabilities JSON"}},
                }
            },
            "/v2/assets": {
                "get": {
                    "summary": "Verify and inspect asset bundle hashes",
                    "responses": {"200": {"description": "Asset verification JSON"}},
                }
            },
            "/v2/goal": {
                "get": {
                    "summary": "Build a scaled left-foreleg goal trajectory",
                    "parameters": [
                        {"name": "scale", "in": "query", "schema": {"type": "number"}, "required": True},
                        {
                            "name": "scope",
                            "in": "query",
                            "schema": {"type": "string", "enum": ["tibia", "lf15"], "default": "tibia"},
                        },
                        {
                            "name": "joints",
                            "in": "query",
                            "description": (
                                "Comma-separated IDs for LF15 kinematic goals. Other columns keep their nominal targets."
                            ),
                            "style": "form",
                            "explode": False,
                            "schema": {"type": "array", "items": {"enum": list(LF_JOINT_NAMES)}, "uniqueItems": True},
                        },
                        {
                            "name": "family",
                            "in": "query",
                            "description": "Optional LF15 goal family. Physics-generated families cover all seven joints.",
                            "schema": {
                                "type": "string",
                                "enum": list(MULTIJOINT_GOAL_FAMILIES),
                                "default": "kinematic_teacher_scaled",
                            },
                        },
                        {"name": "initial_delta_deg", "in": "query", "schema": {"type": "number"}, "required": False},
                    ],
                    "responses": {"200": {"description": "Goal trajectory JSON"}},
                }
            },
            "/v1/goal": {
                "get": {
                    "summary": "V1-compatible goal builder",
                    "responses": {"200": {"description": "Goal trajectory JSON"}},
                }
            },
            "/v2/multijoint/goal": {
                "get": {
                    "summary": "Build a 7-DOF left-foreleg multi-joint goal trajectory from a motion family",
                    "parameters": [
                        {"name": "scale", "in": "query", "schema": {"type": "number", "default": 0.92}},
                        {
                            "name": "family",
                            "in": "query",
                            "schema": {
                                "type": "string",
                                "enum": list(MULTIJOINT_GOAL_FAMILIES),
                                "default": "swing_scaled",
                            },
                        },
                        {
                            "name": "joints",
                            "in": "query",
                            "style": "form",
                            "explode": False,
                            "schema": {"type": "array", "items": {"enum": list(LF_JOINT_NAMES)}, "uniqueItems": True},
                        },
                        {
                            "name": "initial_delta_deg",
                            "in": "query",
                            "description": "Seven comma-separated joint offsets in degrees",
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": {"200": {"description": "Multi-joint goal trajectory JSON"}},
                }
            },
            "/v2/jobs": {
                "post": {
                    "summary": "Submit an async rate-control inference + physics verification job",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "oneOf": [
                                        {"$ref": "#/components/schemas/JobSubmitRequest"},
                                        {"$ref": "#/components/schemas/MultiJointJobSubmitRequest"},
                                    ]
                                }
                            }
                        },
                    },
                    "responses": {
                        "202": {"description": "Queued job descriptor"},
                        "400": {"description": "Invalid request"},
                        "503": {"description": "Requested controller is unavailable"},
                    },
                }
            },
            "/v1/jobs": {
                "post": {
                    "summary": "V1-compatible job submission alias",
                    "responses": {"202": {"description": "Queued job descriptor"}},
                }
            },
            "/v2/multijoint/jobs": {
                "post": {
                    "summary": "Submit an async 15-muscle (0 teacher-assist) 7-DOF LF control job",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MultiJointJobSubmitRequest"},
                            }
                        },
                    },
                    "responses": {
                        "202": {"description": "Queued multi-joint job descriptor"},
                        "400": {"description": "Invalid request"},
                        "503": {"description": "Multi-joint controller is unavailable"},
                    },
                }
            },
            "/v2/bio/jobs": {
                "post": {
                    "summary": "Submit an async biological spike/LIF/MaleCNS premotor realization job",
                    "description": (
                        "Check bio_interface.modes_by_scope in capabilities for usable modes. "
                        "Premotor connectome mode requires local MaleCNS data and a Parquet reader."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/BioJobSubmitRequest"},
                            }
                        },
                    },
                    "responses": {
                        "202": {"description": "Queued biological realization job descriptor"},
                        "400": {"description": "Invalid request"},
                        "404": {"description": "Referenced source_job_id was not found"},
                        "409": {"description": "Referenced source_job_id has not completed yet"},
                    },
                }
            },
            "/v2/jobs/{job_id}": {
                "get": {
                    "summary": "Poll job status, progress, and completed summary",
                    "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {"200": {"description": "Job status JSON"}},
                }
            },
            "/v1/jobs/{job_id}": {
                "get": {
                    "summary": "V1-compatible job polling alias",
                    "responses": {"200": {"description": "Job status JSON"}},
                }
            },
            "/v2/jobs/{job_id}/artifacts/{path}": {
                "get": {
                    "summary": "Fetch a single run artifact (JSON, NPZ, MP4, WebM, PNG)",
                    "parameters": [
                        {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}},
                        {"name": "path", "in": "path", "required": True, "schema": {"type": "string"}},
                        {
                            "name": "Range",
                            "in": "header",
                            "description": "Single byte range, for example bytes=0-1023",
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": {
                        "200": {"description": "Artifact bytes"},
                        "206": {"description": "Requested byte range"},
                        "416": {"description": "Unsupported or unsatisfiable byte range"},
                    },
                }
            },
            "/v1/jobs/{job_id}/artifacts/{path}": {
                "get": {
                    "summary": "V1-compatible artifact download alias",
                    "responses": {"200": {"description": "Artifact bytes"}},
                }
            },
            "/v2/jobs/{job_id}/download": {
                "get": {
                    "summary": "Download non-iteration job artifacts as a ZIP archive",
                    "responses": {"200": {"description": "ZIP bundle"}},
                }
            },
            "/v1/jobs/{job_id}/download": {
                "get": {
                    "summary": "V1-compatible ZIP archive alias",
                    "responses": {"200": {"description": "ZIP bundle"}},
                }
            },
        },
        "components": {
            "schemas": {
                "JobSubmitRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "scope": {"const": "tibia", "default": "tibia"},
                        "goal_scale": {"type": "number", "exclusiveMinimum": 0},
                        "target_rad": {"type": "array", "items": {"type": "number"}, "minItems": 211, "maxItems": 211},
                        "time_s": {"type": "array", "items": {"type": "number"}, "minItems": 211, "maxItems": 211},
                        "mode": {
                            "type": "string",
                            "enum": ["fast", "auto", "refine", "closed_loop"],
                            "default": "auto",
                        },
                        "threshold_deg": {"type": "number", "exclusiveMinimum": 0},
                        "correction_steps": {"type": "integer", "minimum": 0, "maximum": 20},
                        "correction_lr_hz": {"type": "number", "exclusiveMinimum": 0},
                        "correction_optimizer": {
                            "type": "string",
                            "enum": ["lbfgs_linesearch", "adam"],
                            "default": "lbfgs_linesearch",
                        },
                        "feedback_kp": {"type": "number", "minimum": 0, "default": 20.0},
                        "feedback_kd": {"type": "number", "minimum": 0, "default": 0.3},
                        "initial_tibia_delta_deg": {"type": "number", "default": 0.0},
                        "render": {"type": "boolean"},
                        "check_directions": {"type": "boolean", "default": False},
                    },
                },
                "MultiJointJobSubmitRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["scope"],
                    "oneOf": [
                        {"required": ["target_rad"], "not": {"required": ["goal_scale"]}},
                        {
                            "required": ["goal_scale"],
                            "not": {"anyOf": [{"required": ["target_rad"]}, {"required": ["time_s"]}]},
                        },
                    ],
                    "properties": {
                        "scope": {"const": "lf15", "default": "lf15"},
                        "goal_family": {
                            "type": "string",
                            "enum": list(MULTIJOINT_GOAL_FAMILIES),
                            "default": "swing_scaled",
                        },
                        "goal_scale": {"type": "number", "exclusiveMinimum": 0},
                        "target_rad": {
                            "type": "array",
                            "description": "Radians, in time-major order and using the joint order from capabilities.",
                            "minItems": STATE_STEPS,
                            "maxItems": STATE_STEPS,
                            "items": {
                                "type": "array", "items": {"type": "number"},
                                "minItems": len(LF_JOINT_NAMES), "maxItems": len(LF_JOINT_NAMES),
                            },
                        },
                        "time_s": {
                            "type": "array", "items": {"type": "number"},
                            "minItems": STATE_STEPS, "maxItems": STATE_STEPS,
                        },
                        "joint_names": {
                            "type": "array",
                            "description": "Joint IDs selected for the scaled goal and display.",
                            "items": {"enum": list(LF_JOINT_NAMES)}, "minItems": 1, "uniqueItems": True,
                        },
                        "mode": {"enum": ["fast", "auto", "refine", "closed_loop"], "default": "auto"},
                        "threshold_deg": {"type": "number", "exclusiveMinimum": 0},
                        "correction_steps": {"type": "integer", "minimum": 0, "maximum": 20, "default": 4},
                        "feedback_kp": {"type": "number", "minimum": 0, "default": 8.0},
                        "feedback_kd": {"type": "number", "minimum": 0, "default": 0.04},
                        "initial_qpos_delta_deg": {
                            "type": "array", "items": {"type": "number"},
                            "minItems": len(LF_JOINT_NAMES), "maxItems": len(LF_JOINT_NAMES),
                        },
                        "render": {"type": "boolean"},
                        "bio_mode": {
                            "enum": [
                                "none",
                                "deterministic_isi",
                                "poisson_refractory",
                                "direct_lif",
                                "premotor_connectome",
                            ]
                        },
                        "bio_seed": {"type": "integer", "minimum": 0, "maximum": 2**32 - 1, "default": 0},
                        "bio_physics_vjp_steps": {"type": "integer", "minimum": 0, "maximum": 100, "default": 0},
                        "bio_population_scale": {"type": "integer", "minimum": 1, "maximum": 128, "default": 1},
                    },
                },
                "BioJobSubmitRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "scope": {"enum": ["tibia", "lf15"], "default": "lf15"},
                        "bio_mode": {
                            "enum": [
                                "deterministic_isi",
                                "poisson_refractory",
                                "direct_lif",
                                "premotor_connectome",
                            ],
                            "default": "deterministic_isi",
                        },
                        "source_job_id": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
                        "goal_family": {"enum": list(MULTIJOINT_GOAL_FAMILIES), "default": "swing_scaled"},
                        "goal_scale": {"type": "number", "exclusiveMinimum": 0},
                        "command_pool_rates_hz": {
                            "type": "array",
                            "minItems": CONTROL_STEPS * 2,
                            "maxItems": CONTROL_STEPS * 2,
                        },
                        "target_excitation": {
                            "type": "array",
                            "minItems": CONTROL_STEPS,
                            "maxItems": CONTROL_STEPS,
                        },
                        "target_rad": {"type": "array"},
                        "tau_bridge_ms": {"type": "number", "exclusiveMinimum": 0},
                        "population_scale": {"type": "integer", "minimum": 1, "maximum": 128, "default": 1},
                        "premotor_opt_steps": {"type": "integer", "minimum": 0, "maximum": 400, "default": 40},
                        "physics_vjp_steps": {"type": "integer", "minimum": 0, "maximum": 100, "default": 0},
                        "seed": {"type": "integer", "minimum": 0, "maximum": 2**32 - 1, "default": 0},
                    },
                },
            }
        },
    }
