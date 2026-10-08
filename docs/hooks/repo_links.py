"""MkDocs hook: send links that leave docs/ to the file on GitHub.

Pages keep ordinary relative links, which GitHub renders directly when the
repository is browsed. On the site, a link to a repository file outside docs/
(a source README, a research note, an internal development note) would lead
nowhere, so at build time it is rewritten to that file's GitHub URL on the
branch named by `extra.repo_branch` in mkdocs.yml.

Hooks run after plugins, so links in content pulled in by include-markdown are
rewritten too, once that plugin has made them relative to the including page.
"""
import os
import re

LINK = re.compile(r'(\]\()([^)\s]+)(\))')
SCHEME = re.compile(r'^([a-z][a-z0-9+.-]*:|#|/)', re.I)


def on_page_markdown(markdown, page, config, files):
    repo_root = os.path.dirname(os.path.abspath(config['config_file_path']))
    docs_dir = os.path.abspath(config['docs_dir'])
    base = config['repo_url'].rstrip('/')
    branch = config.get('extra', {}).get('repo_branch', 'main')
    page_dir = os.path.dirname(page.file.abs_src_path)

    def rewrite(m):
        target = m.group(2)
        if SCHEME.match(target):
            return m.group(0)
        path, _, anchor = target.partition('#')
        resolved = os.path.normpath(os.path.join(page_dir, path))
        if resolved.startswith(docs_dir + os.sep) or not os.path.exists(resolved):
            return m.group(0)       # a site page, or a genuinely broken link
        rel = os.path.relpath(resolved, repo_root)
        if rel.startswith('..'):
            return m.group(0)
        kind = 'tree' if os.path.isdir(resolved) else 'blob'
        url = f'{base}/{kind}/{branch}/{rel}' + (f'#{anchor}' if anchor else '')
        return m.group(1) + url + m.group(3)

    return LINK.sub(rewrite, markdown)
