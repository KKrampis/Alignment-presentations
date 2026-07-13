const puppeteer = require('./node_modules/puppeteer-core');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({
    executablePath: '/home/claude/.cache/puppeteer/chrome/linux-131.0.6778.204/chrome-linux64/chrome',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'],
    headless: true,
  });
  const page = await browser.newPage();
  const htmlPath = path.resolve(__dirname, 'dist/slides.html');
  await page.goto('file://' + htmlPath, { waitUntil: 'networkidle0', timeout: 30000 });
  await page.pdf({
    path: 'dist/slides.pdf',
    printBackground: true,
    preferCSSPageSize: true,
    timeout: 30000,
  });
  await browser.close();
  console.log('PDF generated: dist/slides.pdf');
})();
