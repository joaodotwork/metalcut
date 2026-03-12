#!/usr/bin/env node
/**
 * Seiva Analyzer - Intelligent context.md population
 *
 * Analyzes git commits and project files to intelligently update context.md
 * Called by the post-commit hook after basic updates are done.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const CONTEXT_FILE = '.claude/context.md';
const MAX_RECENT_COMMITS = 10;

/**
 * Safely execute a git command
 */
function git(cmd) {
  try {
    return execSync(`git ${cmd}`, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }).trim();
  } catch (e) {
    return '';
  }
}

/**
 * Parse a section from context.md
 */
function parseSection(content, sectionName) {
  const regex = new RegExp(`## ${sectionName}\\n([\\s\\S]*?)(?=\\n## |$)`, 'i');
  const match = content.match(regex);
  return match ? match[1].trim() : '';
}

/**
 * Check if a section has only placeholder content
 */
function isPlaceholder(sectionContent) {
  return sectionContent.includes('[') && sectionContent.includes(']') &&
         (sectionContent.includes('Describe') || sectionContent.includes('List') ||
          sectionContent.includes('Important') || sectionContent.includes('Add'));
}

/**
 * Detect tech stack from dependency files
 */
function detectTechStack() {
  const stack = {
    runtime: null,
    framework: [],
    database: [],
    testing: [],
    build: [],
    other: []
  };

  // Node.js / package.json
  if (fs.existsSync('package.json')) {
    try {
      const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
      stack.runtime = 'Node.js';

      const deps = { ...pkg.dependencies, ...pkg.devDependencies };

      // Frameworks
      if (deps.express) stack.framework.push(`Express ${deps.express.replace(/[\^~]/g, '')}`);
      if (deps.fastify) stack.framework.push(`Fastify ${deps.fastify.replace(/[\^~]/g, '')}`);
      if (deps.koa) stack.framework.push(`Koa ${deps.koa.replace(/[\^~]/g, '')}`);
      if (deps.next) stack.framework.push(`Next.js ${deps.next.replace(/[\^~]/g, '')}`);
      if (deps.react) stack.framework.push(`React ${deps.react.replace(/[\^~]/g, '')}`);
      if (deps.vue) stack.framework.push(`Vue ${deps.vue.replace(/[\^~]/g, '')}`);
      if (deps.svelte) stack.framework.push('Svelte');
      if (deps['@angular/core']) stack.framework.push('Angular');

      // Databases
      if (deps.pg || deps.postgres) stack.database.push('PostgreSQL');
      if (deps.mysql || deps.mysql2) stack.database.push('MySQL');
      if (deps.mongodb || deps.mongoose) stack.database.push('MongoDB');
      if (deps.redis || deps.ioredis) stack.database.push('Redis');
      if (deps.sqlite3 || deps['better-sqlite3']) stack.database.push('SQLite');
      if (deps.prisma || deps['@prisma/client']) stack.database.push('Prisma ORM');
      if (deps.sequelize) stack.database.push('Sequelize ORM');
      if (deps.typeorm) stack.database.push('TypeORM');

      // Testing
      if (deps.jest) stack.testing.push('Jest');
      if (deps.mocha) stack.testing.push('Mocha');
      if (deps.vitest) stack.testing.push('Vitest');
      if (deps.playwright || deps['@playwright/test']) stack.testing.push('Playwright');
      if (deps.cypress) stack.testing.push('Cypress');

      // Build tools
      if (deps.typescript) stack.build.push('TypeScript');
      if (deps.vite) stack.build.push('Vite');
      if (deps.webpack) stack.build.push('Webpack');
      if (deps.esbuild) stack.build.push('esbuild');
      if (deps.rollup) stack.build.push('Rollup');
      if (deps.babel || deps['@babel/core']) stack.build.push('Babel');

    } catch (e) {
      // Ignore parse errors
    }
  }

  // Python / requirements.txt or pyproject.toml
  if (fs.existsSync('requirements.txt')) {
    stack.runtime = stack.runtime || 'Python';
    const reqs = fs.readFileSync('requirements.txt', 'utf8');
    if (reqs.includes('django')) stack.framework.push('Django');
    if (reqs.includes('flask')) stack.framework.push('Flask');
    if (reqs.includes('fastapi')) stack.framework.push('FastAPI');
    if (reqs.includes('pytest')) stack.testing.push('pytest');
    if (reqs.includes('sqlalchemy')) stack.database.push('SQLAlchemy');
  }

  if (fs.existsSync('pyproject.toml')) {
    stack.runtime = stack.runtime || 'Python';
    const pyproject = fs.readFileSync('pyproject.toml', 'utf8');
    if (pyproject.includes('django')) stack.framework.push('Django');
    if (pyproject.includes('flask')) stack.framework.push('Flask');
    if (pyproject.includes('fastapi')) stack.framework.push('FastAPI');
    if (pyproject.includes('pytest')) stack.testing.push('pytest');
  }

  // Go / go.mod
  if (fs.existsSync('go.mod')) {
    stack.runtime = stack.runtime || 'Go';
    const gomod = fs.readFileSync('go.mod', 'utf8');
    if (gomod.includes('gin-gonic')) stack.framework.push('Gin');
    if (gomod.includes('echo')) stack.framework.push('Echo');
    if (gomod.includes('fiber')) stack.framework.push('Fiber');
  }

  // Rust / Cargo.toml
  if (fs.existsSync('Cargo.toml')) {
    stack.runtime = stack.runtime || 'Rust';
    const cargo = fs.readFileSync('Cargo.toml', 'utf8');
    if (cargo.includes('actix')) stack.framework.push('Actix');
    if (cargo.includes('rocket')) stack.framework.push('Rocket');
    if (cargo.includes('axum')) stack.framework.push('Axum');
    if (cargo.includes('tokio')) stack.other.push('Tokio');
  }

  // Ruby / Gemfile
  if (fs.existsSync('Gemfile')) {
    stack.runtime = stack.runtime || 'Ruby';
    const gemfile = fs.readFileSync('Gemfile', 'utf8');
    if (gemfile.includes('rails')) stack.framework.push('Rails');
    if (gemfile.includes('sinatra')) stack.framework.push('Sinatra');
    if (gemfile.includes('rspec')) stack.testing.push('RSpec');
  }

  return stack;
}

/**
 * Format tech stack as markdown
 */
function formatTechStack(stack) {
  const lines = [];

  if (stack.runtime) {
    lines.push(`- **Runtime:** ${stack.runtime}`);
  }
  if (stack.framework.length) {
    lines.push(`- **Framework:** ${stack.framework.join(', ')}`);
  }
  if (stack.database.length) {
    lines.push(`- **Database:** ${stack.database.join(', ')}`);
  }
  if (stack.testing.length) {
    lines.push(`- **Testing:** ${stack.testing.join(', ')}`);
  }
  if (stack.build.length) {
    lines.push(`- **Build:** ${stack.build.join(', ')}`);
  }
  if (stack.other.length) {
    lines.push(`- **Other:** ${stack.other.join(', ')}`);
  }

  return lines.length ? lines.join('\n') : null;
}

/**
 * Get frequently modified files from recent commits
 */
function getKeyFiles() {
  const fileChanges = {};

  // Get files changed in recent commits
  const log = git(`log --oneline --name-only -${MAX_RECENT_COMMITS}`);
  if (!log) return null;

  const lines = log.split('\n');
  let currentCommit = null;

  for (const line of lines) {
    if (line.match(/^[a-f0-9]{7,}/)) {
      currentCommit = line;
    } else if (line.trim() && currentCommit) {
      const file = line.trim();
      // Skip certain files
      if (file.includes('context.md') || file.includes('package-lock') ||
          file.includes('yarn.lock') || file.includes('node_modules')) continue;

      fileChanges[file] = (fileChanges[file] || 0) + 1;
    }
  }

  // Sort by frequency and take top files
  const sorted = Object.entries(fileChanges)
    .filter(([_, count]) => count >= 2) // At least 2 changes
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);

  if (!sorted.length) return null;

  const lines2 = sorted.map(([file, count]) => {
    const category = categorizeFile(file);
    return `- \`${file}\` - ${category} (modified ${count}x recently)`;
  });

  return lines2.join('\n');
}

/**
 * Categorize a file by its path/name
 */
function categorizeFile(filepath) {
  const lower = filepath.toLowerCase();
  const base = path.basename(filepath).toLowerCase();

  if (base.includes('config') || base.includes('settings') || base === '.env.example') return 'Configuration';
  if (base === 'index.js' || base === 'index.ts' || base === 'main.js' || base === 'main.ts' || base === 'app.js' || base === 'app.ts') return 'Entry point';
  if (lower.includes('test') || lower.includes('spec')) return 'Tests';
  if (lower.includes('route') || lower.includes('controller')) return 'Routes/Controllers';
  if (lower.includes('model') || lower.includes('schema')) return 'Data models';
  if (lower.includes('util') || lower.includes('helper') || lower.includes('lib/')) return 'Utilities';
  if (lower.includes('component')) return 'UI Component';
  if (lower.includes('hook')) return 'React Hook';
  if (lower.includes('service')) return 'Service';
  if (lower.includes('middleware')) return 'Middleware';
  if (base === 'readme.md' || base === 'changelog.md') return 'Documentation';

  return 'Core logic';
}

/**
 * Analyze recent commits to determine what's being built
 */
function analyzeWorkStreams() {
  const log = git(`log --oneline -${MAX_RECENT_COMMITS}`);
  if (!log) return null;

  const commits = log.split('\n').filter(Boolean);
  const streams = {
    feat: [],
    fix: [],
    refactor: [],
    docs: [],
    test: [],
    chore: [],
    other: []
  };

  for (const commit of commits) {
    const match = commit.match(/^[a-f0-9]+\s+(?:(\w+)(?:\(.*?\))?:\s*)?(.+)$/i);
    if (match) {
      const [, type, msg] = match;
      const lowerType = (type || '').toLowerCase();

      if (lowerType === 'feat' || lowerType === 'feature') {
        streams.feat.push(msg);
      } else if (lowerType === 'fix' || lowerType === 'bugfix') {
        streams.fix.push(msg);
      } else if (lowerType === 'refactor') {
        streams.refactor.push(msg);
      } else if (lowerType === 'docs' || lowerType === 'doc') {
        streams.docs.push(msg);
      } else if (lowerType === 'test' || lowerType === 'tests') {
        streams.test.push(msg);
      } else if (lowerType === 'chore') {
        streams.chore.push(msg);
      } else {
        streams.other.push(msg);
      }
    }
  }

  const lines = [];

  if (streams.feat.length) {
    const summary = summarizeCommits(streams.feat);
    lines.push(`- **Features in progress:** ${summary}`);
  }
  if (streams.fix.length) {
    const summary = summarizeCommits(streams.fix);
    lines.push(`- **Bug fixes:** ${summary}`);
  }
  if (streams.refactor.length) {
    const summary = summarizeCommits(streams.refactor);
    lines.push(`- **Refactoring:** ${summary}`);
  }
  if (streams.test.length) {
    lines.push(`- **Testing:** Adding/updating tests`);
  }

  // If no conventional commits, try to infer from messages
  if (!lines.length && streams.other.length) {
    const summary = summarizeCommits(streams.other);
    lines.push(`- **Recent work:** ${summary}`);
  }

  return lines.length ? lines.join('\n') : null;
}

/**
 * Summarize a list of commit messages into a short description
 */
function summarizeCommits(messages) {
  if (messages.length === 1) return messages[0];

  // Find common words/themes
  const words = {};
  for (const msg of messages) {
    const tokens = msg.toLowerCase().split(/\s+/);
    for (const word of tokens) {
      if (word.length > 3 && !['the', 'and', 'for', 'with', 'from', 'that', 'this'].includes(word)) {
        words[word] = (words[word] || 0) + 1;
      }
    }
  }

  // Return first message with count indicator
  const first = messages[0].length > 50 ? messages[0].slice(0, 47) + '...' : messages[0];
  return `${first} (+${messages.length - 1} more)`;
}

/**
 * Detect active development patterns
 */
function detectPatterns() {
  const patterns = [];

  // Check recent files changed
  const recentFiles = git('diff-tree --no-commit-id --name-only -r HEAD');
  const allRecentFiles = git(`log --oneline --name-only -5`);

  // TDD pattern
  if (allRecentFiles.includes('test') || allRecentFiles.includes('spec')) {
    patterns.push('- Test-driven development active');
  }

  // CI/CD work
  if (allRecentFiles.includes('.github/workflows') || allRecentFiles.includes('.gitlab-ci') ||
      allRecentFiles.includes('Jenkinsfile') || allRecentFiles.includes('.circleci')) {
    patterns.push('- CI/CD pipeline development');
  }

  // Database migrations
  if (allRecentFiles.includes('migration') || allRecentFiles.includes('migrate')) {
    patterns.push('- Database schema evolving');
  }

  // Type definitions
  if (allRecentFiles.includes('.d.ts') || allRecentFiles.includes('types.ts') ||
      allRecentFiles.includes('interfaces.ts')) {
    patterns.push('- Type system refinement');
  }

  // Documentation
  if (allRecentFiles.includes('README') || allRecentFiles.includes('CHANGELOG') ||
      allRecentFiles.includes('docs/')) {
    patterns.push('- Documentation updates');
  }

  // API development
  if (allRecentFiles.includes('route') || allRecentFiles.includes('controller') ||
      allRecentFiles.includes('endpoint') || allRecentFiles.includes('api/')) {
    patterns.push('- API development');
  }

  // Default patterns
  patterns.push('- Session management via Plantas');
  patterns.push('- Context coordination via Seiva');

  return patterns.join('\n');
}

/**
 * Update a section in context.md
 */
function updateSection(content, sectionName, newContent, forceUpdate = false) {
  const currentSection = parseSection(content, sectionName);

  // Only update if section is placeholder OR forceUpdate is true
  if (!forceUpdate && !isPlaceholder(currentSection) && currentSection.length > 0) {
    return content; // Don't overwrite user content
  }

  const regex = new RegExp(`(## ${sectionName}\\n)[\\s\\S]*?(?=\\n## |$)`, 'i');

  if (content.match(regex)) {
    return content.replace(regex, `$1${newContent}\n\n`);
  }

  return content;
}

/**
 * Main analysis function
 */
function analyze() {
  if (!fs.existsSync(CONTEXT_FILE)) {
    return; // No context file, nothing to do
  }

  let content = fs.readFileSync(CONTEXT_FILE, 'utf8');
  let updated = false;

  // 1. Tech Stack - only update if placeholder
  const techStack = detectTechStack();
  const techStackFormatted = formatTechStack(techStack);
  if (techStackFormatted) {
    const before = content;
    content = updateSection(content, 'Tech Stack', techStackFormatted);
    if (content !== before) updated = true;
  }

  // 2. Key Files - only update if placeholder
  const keyFiles = getKeyFiles();
  if (keyFiles) {
    const before = content;
    content = updateSection(content, 'Key Files to Know', keyFiles);
    if (content !== before) updated = true;
  }

  // 3. What's Being Built - only update if placeholder
  const workStreams = analyzeWorkStreams();
  if (workStreams) {
    const before = content;
    content = updateSection(content, "What's Being Built", workStreams);
    if (content !== before) updated = true;
  }

  // 4. Active Patterns - only update if placeholder
  const patterns = detectPatterns();
  if (patterns) {
    const before = content;
    content = updateSection(content, 'Active Patterns', patterns);
    if (content !== before) updated = true;
  }

  // Write back if changes were made
  if (updated) {
    fs.writeFileSync(CONTEXT_FILE, content);
  }
}

// Run if called directly
if (require.main === module) {
  try {
    analyze();
  } catch (e) {
    // Silent failure - don't break commits
    process.exit(0);
  }
}

module.exports = { analyze, detectTechStack, getKeyFiles, analyzeWorkStreams, detectPatterns };
