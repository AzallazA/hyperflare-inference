# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""EXL3 (trellis-coded) routed experts for DeepSeek V4.1 Flash: format reader and CPU reference decoder.

Experimental (branch ``exp/ds41-exl3``). Nothing here is wired into preparation or serving yet; the modules are
torch-free so they run in any Python with NumPy. See ``reference.py`` for the format and its provenance.
"""
