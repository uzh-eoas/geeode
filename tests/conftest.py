#!/usr/bin/env python
# coding: utf-8

# Import and initialize Earth Engine (and other packages required for initialization)
import ee
import json
import os
import pytest
import google.oauth2.credentials

# Initialize Earth Engine using either an auth token (e.g., for CI/CD pipelines) or using the
# pre-existing local authentication. Raises RuntimeError with clear message if neither method succeeds.
def initialize_ee():
    # Method 1: Environment token (for CI / Docker / server deployments)
    token_str = os.getenv("EARTHENGINE_TOKEN")
    if token_str:
        try:
            stored = json.loads(token_str)
            credentials = google.oauth2.credentials.Credentials(
                None,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=stored["client_id"],
                client_secret=stored["client_secret"],
                refresh_token=stored["refresh_token"],
                quota_project_id=stored.get("project"),
            )
            ee.Initialize(credentials=credentials)
            return
        except (json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(
                f"EARTHENGINE_TOKEN invalid: {e}. "
                "Format: JSON with client_id, client_secret, refresh_token, project"
            )

    # Method 2: Local authentication (earthengine authenticate)
    project = os.getenv("EE_PROJECT")
    if not project:
        raise RuntimeError(
            "EE_PROJECT is not set.\n"
            "Set this variable as your Earth Engine (cloud) project ID/name; e.g., 'my-project'.\n"
            "Like so:\n"
            "    EE_PROJECT='my-project' GEEODE_TEST_ASSET_ROOT='projects/my-project/assets' pytest --cov --capture=no\n"
            "See the 'Tests' section of the README for more information."
        )
    try:
        ee.Initialize(project=project)
        return
    except Exception as e:
        raise RuntimeError(
            f"Earth Engine not initialized. Options:\n"
            f"  1) Run 'ee.Authenticate(force=True)' locally via Python, or\n"
            f"  2) Set the EARTHENGINE_TOKEN shell/env var.\n"
            f"Original error: {e}"
        )
